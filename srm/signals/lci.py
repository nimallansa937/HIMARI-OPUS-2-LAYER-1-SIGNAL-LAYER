"""
Leverage Concentration Index (LCI)

Measures open interest concentration across derivatives venues using the
Herfindahl-Hirschman Index (HHI). High concentration means a single venue's
liquidation cascade cannot be absorbed by other venues, amplifying systemic risk.

Forensic Basis:
- October 10, 2025: Binance held 45% of BTC perpetual open interest. When
  Binance's liquidations accelerated, there was no healthy venue to absorb
  the selling pressure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class LCIConfig:
    """
    Configuration for Leverage Concentration Index calculation.
    
    NOTE: Full multi-venue OI data via CoinGlass API is planned but postponed.
    Currently using Binance OI with estimated distribution for other venues.
    Set use_coinglass=True and provide coinglass_api_key when subscription is available.
    """
    hhi_high_threshold: float = 2500  # HHI above this = elevated risk
    hhi_critical_threshold: float = 5000  # HHI above this = critical
    venues: List[str] = field(default_factory=lambda: ['binance', 'bybit', 'okx', 'deribit', 'bitget'])
    use_coinglass: bool = False  # Set to True if you have CoinGlass subscription
    coinglass_api_key: Optional[str] = None
    timeout_seconds: float = 10.0


class LeverageConcentrationIndex:
    """
    Measures open interest concentration across derivatives venues.
    
    The LCI applies the Herfindahl-Hirschman Index (HHI) to open interest
    distribution. High concentration means a single venue's liquidation
    cascade cannot be absorbed by other venues, amplifying systemic risk.
    
    HHI = sum of squared market shares (as percentages)
    - HHI < 1500: Unconcentrated (healthy)
    - HHI 1500-2500: Moderate concentration
    - HHI 2500-5000: High concentration (elevated risk)
    - HHI > 5000: Very high concentration (critical risk)
    """
    
    # NOTE: CoinGlass multi-venue aggregation is planned but currently postponed.
    # Current implementation uses single-venue (Binance) OI as fallback.
    
    BINANCE_OI_ENDPOINT = "https://fapi.binance.com/fapi/v1/openInterest"
    BINANCE_PRICE_ENDPOINT = "https://fapi.binance.com/fapi/v1/ticker/price"
    COINGLASS_OI_ENDPOINT = "https://open-api.coinglass.com/api/pro/v1/futures/openInterest"
    
    def __init__(self, config: Optional[LCIConfig] = None):
        self.config = config or LCIConfig()
    
    async def _fetch_binance_oi(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch open interest from Binance Futures."""
        try:
            # Get OI in contracts
            oi_url = f"{self.BINANCE_OI_ENDPOINT}?symbol={symbol}USDT"
            async with session.get(oi_url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status != 200:
                    return None
                oi_data = await resp.json()
                oi_contracts = float(oi_data['openInterest'])
            
            # Get current price to convert to USD
            price_url = f"{self.BINANCE_PRICE_ENDPOINT}?symbol={symbol}USDT"
            async with session.get(price_url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status != 200:
                    return None
                price_data = await resp.json()
                price = float(price_data['price'])
            
            return oi_contracts * price
            
        except Exception as e:
            logger.debug(f"Binance OI fetch failed: {e}")
            return None
    
    async def _fetch_coinglass_oi(self, session: aiohttp.ClientSession, symbol: str) -> Dict[str, float]:
        """Fetch OI breakdown from CoinGlass (requires paid subscription)."""
        if not self.config.coinglass_api_key:
            return {}
        
        try:
            headers = {"coinglassSecret": self.config.coinglass_api_key}
            url = f"{self.COINGLASS_OI_ENDPOINT}?symbol={symbol}"
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                
                # Parse exchange breakdown
                # Note: Actual response structure may vary - consult CoinGlass docs
                if data.get('success') and data.get('data'):
                    oi_data = data['data']
                    return {
                        item['exchangeName'].lower(): item['openInterest']
                        for item in oi_data.get('list', [])
                        if 'exchangeName' in item and 'openInterest' in item
                    }
        except Exception as e:
            logger.debug(f"CoinGlass OI fetch failed: {e}")
        return {}
    
    async def fetch_oi_distribution(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Fetch open interest from each venue.
        
        Without CoinGlass, only Binance OI is available.
        With CoinGlass subscription, full multi-venue breakdown is returned.
        
        Args:
            symbol: Base asset, e.g., 'BTC', 'ETH'
        
        Returns:
            Dict mapping venue name to open interest in USD
        """
        oi_data = {}
        
        async with aiohttp.ClientSession() as session:
            # If CoinGlass is enabled, try to get full breakdown
            if self.config.use_coinglass and self.config.coinglass_api_key:
                coinglass_data = await self._fetch_coinglass_oi(session, symbol)
                if coinglass_data:
                    return coinglass_data
            
            # Fallback: fetch from Binance only
            binance_oi = await self._fetch_binance_oi(session, symbol)
            if binance_oi is not None:
                oi_data['binance'] = binance_oi
                
                # Without CoinGlass, we can estimate other venues as % of Binance
                # Based on historical averages:
                # Binance: ~40-50%, Bybit: ~20-25%, OKX: ~15-20%, Others: ~10-20%
                # This is a rough estimate - for production use CoinGlass
                if len(oi_data) == 1:
                    logger.debug("Using estimated OI distribution (CoinGlass not available)")
                    # Estimate total market OI assuming Binance is ~45%
                    estimated_total = binance_oi / 0.45
                    oi_data['bybit'] = estimated_total * 0.22
                    oi_data['okx'] = estimated_total * 0.18
                    oi_data['deribit'] = estimated_total * 0.10
                    oi_data['bitget'] = estimated_total * 0.05
        
        return oi_data
    
    def calculate(self, oi_distribution: Dict[str, float]) -> Tuple[float, dict]:
        """
        Calculate LCI using Herfindahl-Hirschman Index.
        
        Args:
            oi_distribution: Dict mapping venue names to open interest in USD
        
        Returns:
            Tuple of (lci_score, metadata_dict)
        """
        if not oi_distribution or len(oi_distribution) < 2:
            return 0.0, {'status': 'insufficient_venues', 'venues': list(oi_distribution.keys())}
        
        total_oi = sum(oi_distribution.values())
        if total_oi == 0:
            return 0.0, {'status': 'zero_oi'}
        
        # Calculate market shares as percentages
        market_shares = {venue: (oi / total_oi) * 100 for venue, oi in oi_distribution.items()}
        
        # HHI = sum of squared market shares
        hhi = sum(share ** 2 for share in market_shares.values())
        
        # Normalize to 0-1 scale
        # HHI ranges from ~2000 (5 equal venues) to 10000 (single venue)
        # Map 2500-5000 to 0.5-1.0, below 2500 to 0-0.5
        if hhi >= self.config.hhi_critical_threshold:
            lci_score = 1.0
        elif hhi >= self.config.hhi_high_threshold:
            # Linear interpolation between high and critical
            lci_score = 0.5 + 0.5 * (hhi - self.config.hhi_high_threshold) / \
                        (self.config.hhi_critical_threshold - self.config.hhi_high_threshold)
        else:
            # Below high threshold
            lci_score = 0.5 * hhi / self.config.hhi_high_threshold
        
        # Identify dominant venue
        dominant_venue = max(market_shares, key=market_shares.get)
        dominant_share = market_shares[dominant_venue]
        
        metadata = {
            'hhi': hhi,
            'market_shares': market_shares,
            'total_oi': total_oi,
            'total_oi_billions': total_oi / 1e9,
            'dominant_venue': dominant_venue,
            'dominant_share': dominant_share,
            'venues_tracked': len(oi_distribution),
            'concentration_level': 'CRITICAL' if hhi > 5000 else 'HIGH' if hhi > 2500 else 'MODERATE' if hhi > 1500 else 'LOW',
            'using_estimates': not self.config.use_coinglass,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return lci_score, metadata
    
    async def refresh_from_api(self, symbol: str = "BTC") -> Tuple[float, dict]:
        """
        Fetch OI distribution and calculate LCI.
        
        Args:
            symbol: Base asset symbol
        
        Returns:
            Tuple of (lci_score, metadata)
        """
        oi_distribution = await self.fetch_oi_distribution(symbol)
        return self.calculate(oi_distribution)
