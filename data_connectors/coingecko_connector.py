"""
CoinGecko API Connector

FREE tier: 10,000 calls/month, 30 calls/min
Covers 10M+ tokens across 700+ exchanges.

Features:
- Price data for any token
- Market cap and volume
- Historical data (1 year free)
- Token metadata
"""

import time
import logging
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger(__name__)


class CoinGeckoConnector:
    """
    CoinGecko FREE API connector.
    
    Rate Limits:
    - 30 calls/minute
    - 10,000 calls/month
    
    Usage:
        cg = CoinGeckoConnector()
        btc = cg.get_price("bitcoin")
        history = cg.get_price_history("bitcoin", days=30)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Common coin ID mappings
    COIN_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'USDT': 'tether',
        'BNB': 'binancecoin',
        'SOL': 'solana',
        'XRP': 'ripple',
        'DOGE': 'dogecoin',
        'ADA': 'cardano',
        'AVAX': 'avalanche-2',
        'MATIC': 'matic-network',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko connector.
        
        Args:
            api_key: Optional API key for higher rate limits (Pro plan)
        """
        self.session = requests.Session()
        self._last_request_time = 0
        self._request_count = 0
        
        if api_key:
            self.session.headers['x-cg-demo-api-key'] = api_key
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Rate-limited request to CoinGecko API."""
        # Enforce 30 calls/min = 2 seconds between calls to be safe
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 2.0:
            time.sleep(2.0 - time_since_last)
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params or {})
        self._last_request_time = time.time()
        self._request_count += 1
        
        if response.status_code == 429:
            logger.warning("Rate limited by CoinGecko, waiting 60s")
            time.sleep(60)
            return self._request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID."""
        symbol_upper = symbol.upper().replace('USDT', '').replace('USD', '')
        return self.COIN_IDS.get(symbol_upper, symbol.lower())
    
    def get_price(
        self, 
        coin_id: str,
        vs_currencies: str = "usd",
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
        include_24hr_change: bool = True
    ) -> Dict[str, Any]:
        """
        Get current price for a coin.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            vs_currencies: Quote currency (e.g., "usd", "btc")
            include_market_cap: Include market cap
            include_24hr_vol: Include 24h volume
            include_24hr_change: Include 24h change
            
        Returns:
            Dict with price data
        """
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currencies,
            'include_market_cap': str(include_market_cap).lower(),
            'include_24hr_vol': str(include_24hr_vol).lower(),
            'include_24hr_change': str(include_24hr_change).lower(),
        }
        
        data = self._request("simple/price", params)
        
        if coin_id not in data:
            raise ValueError(f"Coin not found: {coin_id}")
        
        coin_data = data[coin_id]
        return {
            'coin_id': coin_id,
            'price': coin_data.get(vs_currencies),
            'market_cap': coin_data.get(f'{vs_currencies}_market_cap'),
            'volume_24h': coin_data.get(f'{vs_currencies}_24h_vol'),
            'change_24h_pct': coin_data.get(f'{vs_currencies}_24h_change'),
        }
    
    def get_prices_multi(
        self,
        coin_ids: List[str],
        vs_currencies: str = "usd"
    ) -> Dict[str, Dict]:
        """
        Get prices for multiple coins in one call.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currencies: Quote currency
            
        Returns:
            Dict mapping coin_id -> price data
        """
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': vs_currencies,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
        }
        
        return self._request("simple/price", params)
    
    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get comprehensive coin data.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Dict with full coin metadata
        """
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'false',
            'sparkline': 'false',
        }
        
        data = self._request(f"coins/{coin_id}", params)
        
        market = data.get('market_data', {})
        return {
            'id': data['id'],
            'symbol': data['symbol'],
            'name': data['name'],
            'price_usd': market.get('current_price', {}).get('usd'),
            'market_cap': market.get('market_cap', {}).get('usd'),
            'volume_24h': market.get('total_volume', {}).get('usd'),
            'high_24h': market.get('high_24h', {}).get('usd'),
            'low_24h': market.get('low_24h', {}).get('usd'),
            'price_change_24h_pct': market.get('price_change_percentage_24h'),
            'price_change_7d_pct': market.get('price_change_percentage_7d'),
            'price_change_30d_pct': market.get('price_change_percentage_30d'),
            'ath': market.get('ath', {}).get('usd'),
            'ath_change_pct': market.get('ath_change_percentage', {}).get('usd'),
            'circulating_supply': market.get('circulating_supply'),
            'total_supply': market.get('total_supply'),
            'sentiment_votes_up_pct': data.get('sentiment_votes_up_percentage'),
            'sentiment_votes_down_pct': data.get('sentiment_votes_down_percentage'),
        }
    
    def get_price_history(
        self,
        coin_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            vs_currency: Quote currency
            
        Returns:
            List of OHLC-like data points
        """
        params = {
            'vs_currency': vs_currency,
            'days': days,
        }
        
        data = self._request(f"coins/{coin_id}/market_chart", params)
        
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        market_caps = data.get('market_caps', [])
        
        return [
            {
                'timestamp': prices[i][0],
                'price': prices[i][1],
                'volume': volumes[i][1] if i < len(volumes) else None,
                'market_cap': market_caps[i][1] if i < len(market_caps) else None,
            }
            for i in range(len(prices))
        ]
    
    def get_ohlc(
        self,
        coin_id: str,
        days: int = 7,
        vs_currency: str = "usd"
    ) -> List[Dict[str, Any]]:
        """
        Get OHLC candlestick data.
        
        Args:
            coin_id: CoinGecko coin ID
            days: 1, 7, 14, 30, 90, 180, 365
            vs_currency: Quote currency
            
        Returns:
            List of OHLC candles
        """
        params = {
            'vs_currency': vs_currency,
            'days': days,
        }
        
        data = self._request(f"coins/{coin_id}/ohlc", params)
        
        return [
            {
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
            }
            for candle in data
        ]
    
    def get_trending(self) -> List[Dict]:
        """Get trending coins (top 7)."""
        data = self._request("search/trending")
        return [
            {
                'id': coin['item']['id'],
                'symbol': coin['item']['symbol'],
                'name': coin['item']['name'],
                'market_cap_rank': coin['item'].get('market_cap_rank'),
                'score': coin['item']['score'],
            }
            for coin in data.get('coins', [])
        ]
    
    def get_global_data(self) -> Dict[str, Any]:
        """Get global crypto market data."""
        data = self._request("global")
        global_data = data.get('data', {})
        
        return {
            'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd'),
            'total_volume_24h_usd': global_data.get('total_volume', {}).get('usd'),
            'btc_dominance': global_data.get('market_cap_percentage', {}).get('btc'),
            'eth_dominance': global_data.get('market_cap_percentage', {}).get('eth'),
            'active_cryptocurrencies': global_data.get('active_cryptocurrencies'),
            'markets': global_data.get('markets'),
            'market_cap_change_24h_pct': global_data.get('market_cap_change_percentage_24h_usd'),
        }
    
    def search(self, query: str) -> Dict[str, List]:
        """Search for coins, exchanges, categories."""
        return self._request("search", {'query': query})


# Quick test
if __name__ == "__main__":
    cg = CoinGeckoConnector()
    
    print("Testing CoinGecko Connector...")
    btc = cg.get_price("bitcoin")
    print(f"BTC: ${btc['price']:,.2f} ({btc['change_24h_pct']:+.1f}%)")
    
    eth = cg.get_price("ethereum")
    print(f"ETH: ${eth['price']:,.2f} ({eth['change_24h_pct']:+.1f}%)")
    
    global_data = cg.get_global_data()
    print(f"Total Market Cap: ${global_data['total_market_cap_usd']/1e12:.2f}T")
    print(f"BTC Dominance: {global_data['btc_dominance']:.1f}%")
    
    print("✓ CoinGecko connector working!")
