"""
Validate Live OHLCV Data from Binance

Fetches real data from Binance and runs the OHLCV validation suite.
This verifies your data source is production-ready.

Usage:
    python validate_binance_data.py BTCUSDT 1000
"""

import sys
import logging
import requests
import pandas as pd
from datetime import datetime

from validation.ohlcv_validator import OHLCVValidator, validate_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (1m, 5m, 15m, 1h, etc.)
        limit: Number of candles (max 1000)
        
    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    
    logger.info(f"Fetching {limit} {interval} candles for {symbol} from Binance...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]


def main():
    """Main function."""
    # Parse arguments
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print("=" * 60)
    print("HIMARI OHLCV DATA VALIDATION")
    print("=" * 60)
    print(f"Source: Binance API")
    print(f"Symbol: {symbol}")
    print(f"Candles: {limit}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Fetch data
    try:
        df = fetch_binance_klines(symbol, "1m", limit)
        print(f"✓ Fetched {len(df)} candles")
        print(f"  First: {df['timestamp'].iloc[0]}")
        print(f"  Last:  {df['timestamp'].iloc[-1]}")
        print(f"  Price: ${df['close'].iloc[-1]:,.2f}")
        print()
    except Exception as e:
        print(f"✗ Failed to fetch data: {e}")
        return
    
    # Run validation
    report = validate_ohlcv(df, print_report=True)
    
    # Return code
    if report['overall_passed']:
        print("\n✅ DATA VALIDATED - Ready for production!")
        return 0
    else:
        print("\n❌ DATA REJECTED - Fix issues before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
