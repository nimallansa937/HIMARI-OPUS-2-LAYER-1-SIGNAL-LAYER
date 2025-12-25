"""
Stablecoin Stress Index (SCSI)

Monitors stablecoin health across venues to detect collateral stress.
When stablecoins depeg, collateral values drop, triggering liquidations
even without any price movement in the underlying asset.

Forensic Basis:
- October 10, 2025: USDe traded at $0.65 on Binance while maintaining $1.00 on
  decentralized venues, causing cascading liquidations.
- Terra UST: Collapsed to $0.20 globally, causing cascading liquidations across all venues.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


class StablecoinFailureType(Enum):
    """Type of stablecoin failure detected."""
    NONE = "none"
    VENUE_SPECIFIC = "venue"  # One exchange shows different price
    PROTOCOL_LEVEL = "protocol"  # Global depeg across all venues


@dataclass
class SCSIConfig:
    """Configuration for Stablecoin Stress Index calculation."""
    stablecoins: List[str] = field(default_factory=lambda: ['USDT', 'USDC', 'DAI', 'FDUSD'])
    critical_deviation_threshold: float = 0.05  # 5% = score 1.0
    venue_spread_threshold: float = 0.02  # 2% cross-venue spread = concern
    timeout_seconds: float = 5.0


class StablecoinStressIndex:
    """
    Monitors stablecoin health across venues to detect collateral stress.
    
    The SCSI tracks major stablecoins (USDT, USDC, DAI, FDUSD) for two
    distinct failure modes:
    1. Venue-specific: One exchange shows different price (arbitrage opportunity or trap)
    2. Protocol-level: Global depeg indicating fundamental stablecoin failure
    
    The failure type informs response strategy—venue failures may allow escape,
    protocol failures require aggressive deleveraging.
    """
    
    # CoinGecko IDs for stablecoins
    COINGECKO_IDS = {
        'USDT': 'tether',
        'USDC': 'usd-coin',
        'DAI': 'dai',
        'FDUSD': 'first-digital-usd',
        'USDE': 'ethena-usde',
        'TUSD': 'true-usd',
        'FRAX': 'frax',
        'LUSD': 'liquity-usd',
        'BUSD': 'binance-usd',
    }
    
    def __init__(self, config: Optional[SCSIConfig] = None):
        self.config = config or SCSIConfig()
    
    async def _fetch_coingecko_stablecoin_price(
        self, 
        session: aiohttp.ClientSession, 
        stablecoin: str
    ) -> Optional[float]:
        """Fetch stablecoin price from CoinGecko."""
        try:
            coin_id = self.COINGECKO_IDS.get(stablecoin.upper())
            if not coin_id:
                return None
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if coin_id in data:
                        return data[coin_id]['usd']
        except Exception as e:
            logger.debug(f"CoinGecko stablecoin fetch failed for {stablecoin}: {e}")
        return None
    
    async def _fetch_binance_stablecoin_price(
        self, 
        session: aiohttp.ClientSession, 
        stablecoin: str
    ) -> Optional[float]:
        """Fetch stablecoin price from Binance (using BTC pair for arbitrage)."""
        try:
            # Get stablecoin/USDT price (if not USDT itself)
            if stablecoin.upper() == 'USDT':
                return 1.0  # USDT is the quote currency on Binance
            
            # Try to get the pair price
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={stablecoin.upper()}USDT"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data['price'])
        except Exception as e:
            logger.debug(f"Binance stablecoin fetch failed for {stablecoin}: {e}")
        return None
    
    async def fetch_stablecoin_prices(self, stablecoin: str) -> Dict[str, Optional[float]]:
        """
        Fetch stablecoin prices across venues.
        
        Args:
            stablecoin: Stablecoin symbol, e.g., 'USDT', 'USDC'
        
        Returns:
            Dict mapping venue to price
        """
        prices = {}
        
        async with aiohttp.ClientSession() as session:
            # Fetch from multiple sources
            coingecko_price = await self._fetch_coingecko_stablecoin_price(session, stablecoin)
            binance_price = await self._fetch_binance_stablecoin_price(session, stablecoin)
            
            if coingecko_price is not None:
                prices['coingecko'] = coingecko_price
            if binance_price is not None:
                prices['binance'] = binance_price
        
        return prices
    
    async def calculate(self) -> Tuple[float, dict]:
        """
        Calculate SCSI across all monitored stablecoins.
        
        Returns:
            Tuple of (scsi_score, metadata_dict) where metadata includes
            failure_type indicating whether stress is venue-specific or protocol-level
        """
        protocol_stress = {}
        venue_stress = {}
        all_prices = {}
        
        for stable in self.config.stablecoins:
            prices = await self.fetch_stablecoin_prices(stable)
            valid_prices = {k: v for k, v in prices.items() if v is not None}
            all_prices[stable] = valid_prices
            
            if not valid_prices:
                continue
            
            # Protocol stress: how far is the aggregated price from $1.00?
            mean_price = np.mean(list(valid_prices.values()))
            global_deviation = abs(mean_price - 1.00)
            protocol_stress[stable] = min(global_deviation / self.config.critical_deviation_threshold, 1.0)
            
            # Venue stress: how much spread across venues?
            if len(valid_prices) >= 2:
                price_values = list(valid_prices.values())
                spread = max(price_values) - min(price_values)
                venue_stress[stable] = min(spread / self.config.venue_spread_threshold, 1.0)
            else:
                venue_stress[stable] = 0.0
        
        if not protocol_stress:
            return 0.0, {'status': 'no_stablecoin_data'}
        
        # Overall score = worst case across all stablecoins and both failure modes
        max_protocol = max(protocol_stress.values())
        max_venue = max(venue_stress.values()) if venue_stress else 0.0
        
        # Find worst stablecoin
        worst_protocol_stable = max(protocol_stress, key=protocol_stress.get)
        worst_venue_stable = max(venue_stress, key=venue_stress.get) if venue_stress else None
        
        # Determine failure type
        if max_protocol > max_venue and max_protocol > 0.3:
            failure_type = StablecoinFailureType.PROTOCOL_LEVEL
            worst_stablecoin = worst_protocol_stable
        elif max_venue > max_protocol and max_venue > 0.3:
            failure_type = StablecoinFailureType.VENUE_SPECIFIC
            worst_stablecoin = worst_venue_stable
        else:
            failure_type = StablecoinFailureType.NONE
            worst_stablecoin = worst_protocol_stable
        
        scsi_score = max(max_protocol, max_venue)
        
        # Log critical warning if significant depeg detected
        if scsi_score > 0.5:
            logger.warning(
                f"STABLECOIN STRESS DETECTED: {worst_stablecoin} "
                f"score={scsi_score:.3f} failure_type={failure_type.value}"
            )
        
        metadata = {
            'protocol_stress': protocol_stress,
            'venue_stress': venue_stress,
            'max_protocol': max_protocol,
            'max_venue': max_venue,
            'failure_type': failure_type.value,
            'worst_stablecoin': worst_stablecoin,
            'all_prices': all_prices,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return scsi_score, metadata
    
    async def refresh_from_api(self) -> Tuple[float, dict]:
        """
        Calculate current SCSI from API data.
        
        Returns:
            Tuple of (scsi_score, metadata)
        """
        return await self.calculate()
