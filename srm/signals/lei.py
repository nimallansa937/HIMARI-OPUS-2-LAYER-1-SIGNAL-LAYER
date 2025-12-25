"""
Liquidity Evaporation Index (LEI)

Measures order book depth erosion relative to rolling baseline.
When depth thins, even modest sell pressure can move prices dramatically,
triggering liquidations that further deplete depth in a vicious cycle.

Forensic Basis:
- October 10, 2025: 1% depth collapsed from $1.2M to $27K (97.75% evaporation)
- April 18, 2021: Weekend cascade when liquidity was structurally 30% thinner
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class LEIConfig:
    """Configuration for Liquidity Evaporation Index calculation."""
    depth_percentage: float = 0.01  # Measure depth within 1% of mid price
    baseline_window_days: int = 7  # Rolling baseline period
    weekend_adjustment: float = 0.70  # Weekends have 30% less liquidity structurally
    velocity_lookback_hours: int = 1  # Window for depth change velocity
    velocity_penalty_threshold: float = -0.05  # 5% depth decline per hour
    velocity_penalty_magnitude: float = 0.30
    # API endpoint (configurable for testnet/proxy support)
    binance_orderbook_endpoint: str = "https://fapi.binance.com/fapi/v1/depth"


class LiquidityEvaporationIndex:
    """
    Measures order book depth erosion relative to rolling baseline.
    
    The LEI compares current 1% order book depth against a 7-day rolling
    average, adjusted for weekend effects. A velocity component detects
    rapid depth deterioration that often precedes cascade acceleration.
    """
    
    def __init__(self, config: Optional[LEIConfig] = None):
        self.config = config or LEIConfig()
        # Store (timestamp, depth) tuples for baseline calculation
        # 1-minute resolution, 7 days max
        self.depth_history: deque = deque(maxlen=7 * 24 * 60)
        self.hourly_depths: deque = deque(maxlen=24)  # For velocity calculation
    
    def update(self, bid_depth: float, ask_depth: float, timestamp: datetime) -> None:
        """
        Record new order book depth observation.
        
        Args:
            bid_depth: Total bid volume within 1% of mid price (in quote currency, e.g., USD)
            ask_depth: Total ask volume within 1% of mid price
            timestamp: Observation timestamp
        """
        total_depth = bid_depth + ask_depth
        self.depth_history.append((timestamp, total_depth))
        
        # Update hourly snapshots for velocity calculation
        if len(self.hourly_depths) == 0 or \
           (timestamp - self.hourly_depths[-1][0]).total_seconds() >= 3600:
            self.hourly_depths.append((timestamp, total_depth))
    
    def calculate(self, current_depth: float, timestamp: datetime) -> Tuple[float, dict]:
        """
        Calculate LEI with baseline comparison and velocity detection.
        
        Args:
            current_depth: Current total 1% depth (bid + ask)
            timestamp: Current timestamp
        
        Returns:
            Tuple of (lei_score, metadata_dict)
        """
        if len(self.depth_history) < 24 * 60:  # Need at least 1 day of data
            return 0.0, {'status': 'insufficient_baseline', 'observations': len(self.depth_history)}
        
        # Calculate 7-day baseline
        depths = [d[1] for d in self.depth_history]
        baseline_depth = np.mean(depths)
        
        # Weekend adjustment: if today is Sat/Sun, reduce baseline expectation
        is_weekend = timestamp.weekday() >= 5
        adjusted_baseline = baseline_depth
        if is_weekend:
            adjusted_baseline = baseline_depth * self.config.weekend_adjustment
            logger.debug(f"LEI weekend adjustment applied: baseline {baseline_depth:.0f} -> {adjusted_baseline:.0f} ({self.config.weekend_adjustment:.0%})")
        
        # Evaporation score: how much depth has disappeared vs baseline?
        if adjusted_baseline > 0:
            evaporation = 1.0 - (current_depth / adjusted_baseline)
            evaporation = max(0, min(1, evaporation))  # Clamp to 0-1
        else:
            evaporation = 1.0  # No baseline = maximum risk
        
        # Velocity component: is depth declining rapidly?
        velocity_penalty = 0.0
        avg_velocity = 0.0
        if len(self.hourly_depths) >= 2:
            recent_depths = [d[1] for d in list(self.hourly_depths)[-6:]]  # Last 6 hours
            if len(recent_depths) >= 2:
                depth_changes = np.diff(recent_depths) / np.array(recent_depths[:-1])
                avg_velocity = np.mean(depth_changes)
                
                if avg_velocity < self.config.velocity_penalty_threshold:
                    velocity_penalty = self.config.velocity_penalty_magnitude
        
        lei_score = min(evaporation + velocity_penalty, 1.0)
        
        metadata = {
            'current_depth': current_depth,
            'baseline_depth': baseline_depth,
            'adjusted_baseline': adjusted_baseline,
            'is_weekend': is_weekend,
            'evaporation_component': evaporation,
            'evaporation_pct': evaporation * 100,
            'velocity': avg_velocity,
            'velocity_penalty': velocity_penalty,
            'baseline_observations': len(self.depth_history),
            'hourly_observations': len(self.hourly_depths)
        }
        
        return lei_score, metadata
    
    async def fetch_order_book_depth(
        self, 
        symbol: str = "BTCUSDT", 
        limit: int = 500
    ) -> dict:
        """
        Fetch order book and calculate 1% depth.
        
        Args:
            symbol: Trading pair
            limit: Order book depth limit
        
        Returns:
            Dict with 'bid_depth', 'ask_depth', 'mid_price', 'timestamp'
        """
        params = {"symbol": symbol, "limit": limit}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.binance_orderbook_endpoint, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Binance order book API error: {resp.status}")
                        return {'bid_depth': 0, 'ask_depth': 0, 'mid_price': 0, 'timestamp': datetime.utcnow()}
                    
                    data = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'mid_price': 0, 'timestamp': datetime.utcnow()}
        
        # Parse bids and asks
        bids = [(float(price), float(qty)) for price, qty in data.get('bids', [])]
        asks = [(float(price), float(qty)) for price, qty in data.get('asks', [])]
        
        if not bids or not asks:
            return {'bid_depth': 0, 'ask_depth': 0, 'mid_price': 0, 'timestamp': datetime.utcnow()}
        
        mid_price = (bids[0][0] + asks[0][0]) / 2
        price_threshold = mid_price * self.config.depth_percentage  # 1% from mid
        
        # Sum volume within 1% of mid (converted to quote currency)
        bid_depth = sum(price * qty for price, qty in bids if price >= mid_price - price_threshold)
        ask_depth = sum(price * qty for price, qty in asks if price <= mid_price + price_threshold)
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'mid_price': mid_price,
            'total_depth': bid_depth + ask_depth,
            'timestamp': datetime.utcnow()
        }
    
    async def refresh_from_api(self, symbol: str = "BTCUSDT") -> Tuple[float, dict]:
        """
        Fetch current order book and calculate LEI.
        
        Args:
            symbol: Trading pair
        
        Returns:
            Tuple of (lei_score, metadata)
        """
        depth_data = await self.fetch_order_book_depth(symbol)
        
        if depth_data['total_depth'] == 0:
            return 0.0, {'status': 'fetch_failed'}
        
        # Update history
        self.update(
            depth_data['bid_depth'], 
            depth_data['ask_depth'], 
            depth_data['timestamp']
        )
        
        # Calculate score
        return self.calculate(depth_data['total_depth'], depth_data['timestamp'])
    
    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            'depth_history': [(ts.isoformat(), depth) for ts, depth in self.depth_history],
            'hourly_depths': [(ts.isoformat(), depth) for ts, depth in self.hourly_depths]
        }
    
    def restore_state(self, state: dict) -> None:
        """Restore state from persistence."""
        self.depth_history.clear()
        for ts_str, depth in state.get('depth_history', []):
            self.depth_history.append((datetime.fromisoformat(ts_str), depth))
        
        self.hourly_depths.clear()
        for ts_str, depth in state.get('hourly_depths', []):
            self.hourly_depths.append((datetime.fromisoformat(ts_str), depth))
