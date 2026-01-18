"""
CoinCap API Connector (v3 Pro)

Now using CoinCap Pro v3 API with bearer token authentication.
Previously used api.coincap.io/v2 which had DNS issues.
Now uses rest.coincap.io/v3 which is fully operational.

Features:
- Real-time prices for 2000+ assets
- Historical data
- Exchange rates
- WebSocket streaming
- Bearer token authentication for Pro features
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CoinCapConnector:
    """
    CoinCap FREE API connector.
    
    Rate Limits: 200 requests/minute (generous)
    
    Usage:
        cc = CoinCapConnector()
        btc = cc.get_asset("bitcoin")
        history = cc.get_history("bitcoin", interval="d1")
    """
    
    BASE_URL = "https://rest.coincap.io/v3"  # v3 Pro API (v2 was api.coincap.io)
    WS_URL = "wss://ws.coincap.io/prices"
    
    # Symbol to ID mapping
    ASSET_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'USDT': 'tether',
        'BNB': 'binance-coin',
        'SOL': 'solana',
        'XRP': 'xrp',
        'DOGE': 'dogecoin',
        'ADA': 'cardano',
        'AVAX': 'avalanche',
        'MATIC': 'polygon',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinCap connector.
        
        Args:
            api_key: CoinCap Pro API key (loaded from COINCAP_API_KEY env if not provided)
        """
        self.session = requests.Session()
        self._last_request_time = 0
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('COINCAP_API_KEY')
        
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
            logger.info("CoinCap Pro API initialized with bearer token")
        else:
            logger.warning("No COINCAP_API_KEY found - some features may be limited")
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Rate-limited request to CoinCap API."""
        # 200 req/min = 300ms between requests
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 0.3:
            time.sleep(0.3 - time_since_last)
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params or {})
        self._last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning("Rate limited by CoinCap, waiting 60s")
            time.sleep(60)
            return self._request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        """
        Get asset data by ID.
        
        Args:
            asset_id: CoinCap asset ID (e.g., "bitcoin")
            
        Returns:
            Dict with asset data
        """
        data = self._request(f"assets/{asset_id}")
        asset = data.get('data', {})
        
        return {
            'id': asset.get('id'),
            'rank': int(asset.get('rank', 0)),
            'symbol': asset.get('symbol'),
            'name': asset.get('name'),
            'supply': float(asset.get('supply', 0)),
            'max_supply': float(asset.get('maxSupply', 0)) if asset.get('maxSupply') else None,
            'market_cap_usd': float(asset.get('marketCapUsd', 0)),
            'volume_24h_usd': float(asset.get('volumeUsd24Hr', 0)),
            'price_usd': float(asset.get('priceUsd', 0)),
            'change_24h_pct': float(asset.get('changePercent24Hr', 0)),
            'vwap_24h': float(asset.get('vwap24Hr', 0)) if asset.get('vwap24Hr') else None,
        }
    
    def get_assets(
        self,
        search: Optional[str] = None,
        ids: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get multiple assets.
        
        Args:
            search: Search query
            ids: Specific asset IDs
            limit: Max results (default 100)
            offset: Pagination offset
            
        Returns:
            List of asset data
        """
        params = {'limit': limit, 'offset': offset}
        if search:
            params['search'] = search
        if ids:
            params['ids'] = ','.join(ids)
        
        data = self._request("assets", params)
        
        return [
            {
                'id': asset.get('id'),
                'rank': int(asset.get('rank', 0)),
                'symbol': asset.get('symbol'),
                'name': asset.get('name'),
                'price_usd': float(asset.get('priceUsd', 0)),
                'market_cap_usd': float(asset.get('marketCapUsd', 0)),
                'volume_24h_usd': float(asset.get('volumeUsd24Hr', 0)),
                'change_24h_pct': float(asset.get('changePercent24Hr', 0)),
            }
            for asset in data.get('data', [])
        ]
    
    def get_history(
        self,
        asset_id: str,
        interval: str = "d1",
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.
        
        Args:
            asset_id: CoinCap asset ID
            interval: m1, m5, m15, m30, h1, h2, h6, h12, d1
            start: Start timestamp (ms)
            end: End timestamp (ms)
            
        Returns:
            List of price points
        """
        params = {'interval': interval}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        data = self._request(f"assets/{asset_id}/history", params)
        
        return [
            {
                'timestamp': point.get('time'),
                'price_usd': float(point.get('priceUsd', 0)),
                'date': point.get('date'),
            }
            for point in data.get('data', [])
        ]
    
    def get_markets(
        self,
        asset_id: Optional[str] = None,
        exchange_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get market/trading pair data.
        
        Args:
            asset_id: Filter by asset
            exchange_id: Filter by exchange
            limit: Max results
            
        Returns:
            List of market data
        """
        params = {'limit': limit}
        if asset_id:
            params['baseId'] = asset_id
        if exchange_id:
            params['exchangeId'] = exchange_id
        
        data = self._request("markets", params)
        
        return [
            {
                'exchange_id': market.get('exchangeId'),
                'base_id': market.get('baseId'),
                'quote_id': market.get('quoteId'),
                'base_symbol': market.get('baseSymbol'),
                'quote_symbol': market.get('quoteSymbol'),
                'volume_24h_usd': float(market.get('volumeUsd24Hr', 0)),
                'price_usd': float(market.get('priceUsd', 0)),
                'volume_pct': float(market.get('volumePercent', 0)),
            }
            for market in data.get('data', [])
        ]
    
    def get_exchanges(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get exchange data."""
        data = self._request("exchanges", {'limit': limit})
        
        return [
            {
                'id': ex.get('exchangeId'),
                'name': ex.get('name'),
                'rank': int(ex.get('rank', 0)),
                'volume_24h_usd': float(ex.get('volumeUsd', 0)),
                'trading_pairs': int(ex.get('tradingPairs', 0)),
                'percent_volume': float(ex.get('percentTotalVolume', 0)),
            }
            for ex in data.get('data', [])
        ]
    
    def get_rates(self) -> Dict[str, Dict]:
        """Get conversion rates for all currencies."""
        data = self._request("rates")
        
        return {
            rate.get('id'): {
                'symbol': rate.get('symbol'),
                'currency_symbol': rate.get('currencySymbol'),
                'rate_usd': float(rate.get('rateUsd', 0)),
                'type': rate.get('type'),
            }
            for rate in data.get('data', [])
        }


# Quick test
if __name__ == "__main__":
    cc = CoinCapConnector()
    
    print("Testing CoinCap Connector...")
    btc = cc.get_asset("bitcoin")
    print(f"BTC: ${btc['price_usd']:,.2f} (Rank #{btc['rank']})")
    
    top_10 = cc.get_assets(limit=10)
    print(f"Top 10 assets: {[a['symbol'] for a in top_10]}")
    
    history = cc.get_history("bitcoin", interval="d1")
    print(f"Got {len(history)} historical data points")
    
    print("✓ CoinCap connector working!")
