"""
Oracle Divergence Score (ODS)

Detects price divergence across venues and oracle feeds.
Cross-venue price divergence indicates either a data feed failure,
a liquidity crisis on specific venues, or potential manipulation.

Forensic Basis:
- October 10, 2025: USDe traded at $1.00 on Curve (DEX) while showing $0.65 on
  Binance (CEX). Binance used its internal $0.65 price for collateral valuation,
  triggering liquidations that wouldn't have occurred under the "true" oracle price.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ODSConfig:
    """Configuration for Oracle Divergence Score calculation."""
    critical_divergence_threshold: float = 0.05  # 5% divergence = score 1.0
    alert_divergence_threshold: float = 0.10  # Log critical warning above this
    venues: List[str] = field(default_factory=lambda: ['binance', 'coinbase', 'kraken', 'coingecko'])
    timeout_seconds: float = 5.0  # API timeout


class OracleDivergenceScore:
    """
    Detects price divergence across venues and oracle feeds.
    
    Cross-venue price divergence indicates either a data feed failure,
    a liquidity crisis on specific venues, or potential manipulation.
    The ODS compares prices from multiple CEXs against aggregated
    oracle/index prices to detect these dislocations early.
    """
    
    def __init__(self, config: Optional[ODSConfig] = None):
        self.config = config or ODSConfig()
    
    async def _fetch_binance_price(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch price from Binance."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data['price'])
        except Exception as e:
            logger.debug(f"Binance price fetch failed: {e}")
        return None
    
    async def _fetch_coinbase_price(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch price from Coinbase."""
        try:
            base = symbol.replace('USDT', '').replace('USD', '')
            url = f"https://api.coinbase.com/v2/prices/{base}-USD/spot"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data['data']['amount'])
        except Exception as e:
            logger.debug(f"Coinbase price fetch failed: {e}")
        return None
    
    async def _fetch_kraken_price(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch price from Kraken."""
        try:
            base = symbol.replace('USDT', '').replace('USD', '')
            # Kraken uses different symbol formats
            if base == 'BTC':
                kraken_symbol = 'XXBTZUSD'
            elif base == 'ETH':
                kraken_symbol = 'XETHZUSD'
            else:
                kraken_symbol = f"{base}USD"
            
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('error') and len(data['error']) > 0:
                        return None
                    result_key = list(data['result'].keys())[0]
                    return float(data['result'][result_key]['c'][0])  # Last trade close
        except Exception as e:
            logger.debug(f"Kraken price fetch failed: {e}")
        return None
    
    async def _fetch_coingecko_price(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch price from CoinGecko (aggregated oracle-like feed)."""
        try:
            base = symbol.replace('USDT', '').replace('USD', '').lower()
            # Map common symbols to CoinGecko IDs
            coin_id_map = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'bnb': 'binancecoin',
                'xrp': 'ripple',
                'ada': 'cardano',
                'doge': 'dogecoin',
                'avax': 'avalanche-2',
                'dot': 'polkadot',
                'matic': 'matic-network',
            }
            coin_id = coin_id_map.get(base, base)
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if coin_id in data:
                        return data[coin_id]['usd']
        except Exception as e:
            logger.debug(f"CoinGecko price fetch failed: {e}")
        return None
    
    async def fetch_all_prices(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Fetch prices from all configured venues.
        
        Args:
            symbol: Trading pair, e.g., 'BTCUSDT' or 'BTC'
        
        Returns:
            Dict mapping venue name to price, None if fetch failed
        """
        prices = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            venue_names = []
            
            if 'binance' in self.config.venues:
                tasks.append(self._fetch_binance_price(session, symbol))
                venue_names.append('binance')
            
            if 'coinbase' in self.config.venues:
                tasks.append(self._fetch_coinbase_price(session, symbol))
                venue_names.append('coinbase')
            
            if 'kraken' in self.config.venues:
                tasks.append(self._fetch_kraken_price(session, symbol))
                venue_names.append('kraken')
            
            if 'coingecko' in self.config.venues:
                tasks.append(self._fetch_coingecko_price(session, symbol))
                venue_names.append('coingecko')
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for venue, result in zip(venue_names, results):
                if isinstance(result, Exception):
                    prices[venue] = None
                else:
                    prices[venue] = result
        
        return prices
    
    def calculate(self, prices: Dict[str, Optional[float]]) -> Tuple[float, dict]:
        """
        Calculate ODS from multi-venue prices.
        
        Args:
            prices: Dict mapping venue names to prices (None for failed fetches)
        
        Returns:
            Tuple of (ods_score, metadata_dict)
        """
        # Filter out failed fetches
        valid_prices = {k: v for k, v in prices.items() if v is not None}
        
        if len(valid_prices) < 2:
            return 0.0, {'status': 'insufficient_venues', 'valid_venues': list(valid_prices.keys())}
        
        price_values = list(valid_prices.values())
        mean_price = np.mean(price_values)
        
        # Calculate max deviation from mean
        deviations = {venue: abs(price - mean_price) / mean_price 
                      for venue, price in valid_prices.items()}
        max_deviation = max(deviations.values())
        worst_venue = max(deviations, key=deviations.get)
        
        # Calculate pairwise spread (max - min) / mean
        price_spread = (max(price_values) - min(price_values)) / mean_price
        
        # Score based on larger of max deviation or spread
        divergence = max(max_deviation, price_spread)
        ods_score = min(divergence / self.config.critical_divergence_threshold, 1.0)
        
        # Log critical warning if divergence is extreme
        if divergence > self.config.alert_divergence_threshold:
            logger.critical(
                f"EXTREME ORACLE DIVERGENCE: {divergence*100:.2f}% "
                f"(worst: {worst_venue}={valid_prices[worst_venue]:.2f}, mean={mean_price:.2f})"
            )
        
        metadata = {
            'valid_venues': list(valid_prices.keys()),
            'prices': valid_prices,
            'mean_price': mean_price,
            'max_deviation': max_deviation,
            'max_deviation_pct': max_deviation * 100,
            'worst_venue': worst_venue,
            'price_spread': price_spread,
            'price_spread_pct': price_spread * 100,
            'divergence_used': divergence,
            'divergence_pct': divergence * 100,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return ods_score, metadata
    
    async def refresh_from_api(self, symbol: str = "BTC") -> Tuple[float, dict]:
        """
        Fetch prices from all venues and calculate ODS.
        
        Args:
            symbol: Base asset symbol
        
        Returns:
            Tuple of (ods_score, metadata)
        """
        prices = await self.fetch_all_prices(symbol)
        return self.calculate(prices)
