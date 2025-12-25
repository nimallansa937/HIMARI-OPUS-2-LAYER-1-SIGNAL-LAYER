"""
Funding Saturation Index (FSI)

Monitors perpetual futures funding rates to detect overcrowded directional positioning.
Sustained elevated funding rates (>0.05-0.10%) for 48+ hours signals dangerous leverage
accumulation that often precedes liquidation cascades.

Forensic Basis:
- May 19, 2021: Funding rates exceeded 0.15% for multiple days before cascade
- August 5, 2024: Sustained 0.08% rates before yen carry unwind
- October 10, 2025: 0.14% funding sustained 48h before collapse
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class FSIConfig:
    """Configuration for Funding Saturation Index calculation."""
    ema_span: int = 48  # Hours for EMA smoothing
    low_vol_threshold: float = 0.10  # Funding threshold in low volatility regime (as %)
    high_vol_threshold: float = 0.15  # Funding threshold in high volatility regime (as %)
    velocity_lookback: int = 24  # Hours for velocity calculation
    velocity_penalty_threshold: float = 0.01  # %/8h slope triggering penalty
    velocity_penalty_magnitude: float = 0.20  # Added to score when velocity exceeds threshold
    # API endpoint (configurable for testnet/proxy support)
    binance_funding_endpoint: str = "https://fapi.binance.com/fapi/v1/fundingRate"


class FundingSaturationIndex:
    """
    Calculates funding rate saturation with velocity-based acceleration detection.
    
    The FSI measures how close current funding rates are to historically dangerous
    levels, adjusted for market volatility regime. A funding rate of 0.10% in a
    calm market is more dangerous than 0.10% during a volatility spike, because
    calm markets have less natural position turnover.
    """
    
    def __init__(self, config: Optional[FSIConfig] = None):
        self.config = config or FSIConfig()
        self.funding_history: List[float] = []
        self.timestamps: List[datetime] = []
    
    def update(self, funding_rate: float, timestamp: datetime) -> None:
        """
        Append new funding rate observation.
        
        Args:
            funding_rate: Current 8-hour funding rate as decimal (0.0001 = 0.01%)
            timestamp: Observation timestamp
        """
        self.funding_history.append(funding_rate)
        self.timestamps.append(timestamp)
        
        # Maintain 7-day rolling window (21 observations at 8h intervals)
        max_history = 21 * 3  # Extra buffer for robustness
        if len(self.funding_history) > max_history:
            self.funding_history = self.funding_history[-max_history:]
            self.timestamps = self.timestamps[-max_history:]
    
    def calculate(self, volatility_regime: str) -> Tuple[float, dict]:
        """
        Calculate FSI with regime-aware thresholds.
        
        Args:
            volatility_regime: One of 'LOW', 'MEDIUM', 'HIGH' based on VIX or 
                               realized volatility. Use 'LOW' when VIX < 20,
                               'HIGH' when VIX > 30, 'MEDIUM' otherwise.
        
        Returns:
            Tuple of (fsi_score, metadata_dict)
            - fsi_score: Float 0.0-1.0, higher = more risk
            - metadata: Dict with intermediate calculations for debugging
        """
        if len(self.funding_history) < 6:  # Need at least 48h of data
            return 0.0, {'status': 'insufficient_data', 'observations': len(self.funding_history)}
        
        # Convert to pandas for EMA calculation
        funding_series = pd.Series(self.funding_history)
        
        # 48-hour EMA smooths out noise while preserving trend
        # Span is in observations (8h each), so divide by 8
        ema_funding = funding_series.ewm(span=self.config.ema_span // 8).mean().iloc[-1]
        
        # Dynamic threshold based on volatility regime
        # Rationale: In calm markets, traders are complacent and leverage accumulates.
        # In volatile markets, positions turn over faster and extreme funding is more
        # likely to be transient.
        if volatility_regime == 'LOW':
            threshold = self.config.low_vol_threshold / 100  # Convert from % to decimal
        elif volatility_regime == 'HIGH':
            threshold = self.config.high_vol_threshold / 100
        else:  # MEDIUM
            threshold = (self.config.low_vol_threshold + self.config.high_vol_threshold) / 200
        
        # Base saturation score: how close is funding to dangerous threshold?
        saturation_score = min(max(abs(ema_funding) / threshold, 0), 1.0)
        
        # Velocity component: is funding accelerating?
        # The forensic data shows that rising funding is more dangerous than stable
        # elevated funding—it indicates fresh leverage entering, not stale positions.
        velocity_lookback_obs = self.config.velocity_lookback // 8  # Convert hours to 8h observations
        if len(self.funding_history) >= velocity_lookback_obs:
            recent_funding = self.funding_history[-velocity_lookback_obs:]
            slope = np.gradient(recent_funding).mean()
        else:
            slope = 0.0
        
        # Apply velocity penalty if funding is accelerating
        velocity_penalty = self.config.velocity_penalty_magnitude if slope > self.config.velocity_penalty_threshold / 100 else 0.0
        
        # Final score with cap at 1.0
        fsi_score = min(saturation_score + velocity_penalty, 1.0)
        
        metadata = {
            'ema_funding': ema_funding,
            'ema_funding_pct': ema_funding * 100,  # For display
            'threshold_used': threshold,
            'threshold_pct': threshold * 100,
            'volatility_regime': volatility_regime,
            'saturation_component': saturation_score,
            'slope': slope,
            'velocity_penalty': velocity_penalty,
            'observations_used': len(self.funding_history),
            'direction': 'LONG_HEAVY' if ema_funding > 0 else 'SHORT_HEAVY' if ema_funding < 0 else 'NEUTRAL'
        }
        
        return fsi_score, metadata
    
    async def fetch_funding_history(
        self, 
        symbol: str = "BTCUSDT", 
        limit: int = 100
    ) -> List[dict]:
        """
        Fetch historical funding rates from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of funding rate records to fetch
        
        Returns:
            List of dicts with keys: timestamp, rate
        """
        params = {"symbol": symbol, "limit": limit}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.binance_funding_endpoint, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Binance funding API error: {resp.status}")
                        return []
                    
                    data = await resp.json()
                    return [
                        {
                            "timestamp": datetime.fromtimestamp(item["fundingTime"] / 1000),
                            "rate": float(item["fundingRate"])
                        }
                        for item in data
                    ]
        except Exception as e:
            logger.error(f"Failed to fetch funding history: {e}")
            return []
    
    async def refresh_from_api(self, symbol: str = "BTCUSDT") -> bool:
        """
        Refresh funding history from Binance API and update internal state.
        
        Args:
            symbol: Trading pair
        
        Returns:
            True if successful, False otherwise
        """
        history = await self.fetch_funding_history(symbol)
        
        if not history:
            return False
        
        # Clear and repopulate
        self.funding_history.clear()
        self.timestamps.clear()
        
        for entry in history:
            self.update(entry['rate'], entry['timestamp'])
        
        logger.info(f"FSI refreshed with {len(history)} funding rate observations")
        return True
    
    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            'funding_history': self.funding_history.copy(),
            'timestamps': [ts.isoformat() for ts in self.timestamps]
        }
    
    def restore_state(self, state: dict) -> None:
        """Restore state from persistence."""
        self.funding_history = state.get('funding_history', [])
        self.timestamps = [
            datetime.fromisoformat(ts) for ts in state.get('timestamps', [])
        ]
