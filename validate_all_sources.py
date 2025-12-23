"""
Validate All HIMARI Data Sources

Tests and validates data from:
- Binance (primary, free)
- Kraken (backup, free)
- CoinGecko (supplemental, free tier)
- CoinCap (backup, free)

Usage:
    python validate_all_sources.py
"""

import sys
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

# Add parent to path for imports
sys.path.insert(0, '.')

from validation.ohlcv_validator import OHLCVValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean output
)
logger = logging.getLogger(__name__)


class DataSourceValidator:
    """Validate multiple OHLCV data sources."""
    
    def __init__(self):
        self.validator = OHLCVValidator()
        self.results = {}
    
    # =========================================================================
    # BINANCE
    # =========================================================================
    
    def fetch_binance(self, symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
        """Fetch from Binance REST API."""
        url = "https://api.binance.com/api/v3/klines"
        response = requests.get(url, params={
            'symbol': symbol,
            'interval': '1m',
            'limit': limit,
        })
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # =========================================================================
    # KRAKEN
    # =========================================================================
    
    def fetch_kraken(self, symbol: str = "XXBTZUSD", limit: int = 720) -> pd.DataFrame:
        """Fetch from Kraken REST API."""
        url = "https://api.kraken.com/0/public/OHLC"
        response = requests.get(url, params={
            'pair': symbol,
            'interval': 1,  # 1 minute
        })
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('error'):
            raise Exception(f"Kraken error: {data['error']}")
        
        # Kraken returns data under pair name
        result_key = list(data['result'].keys())[0]
        if result_key == 'last':
            result_key = list(data['result'].keys())[1]
        
        ohlc = data['result'][result_key]
        
        df = pd.DataFrame(ohlc, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'vwap', 'volume', 'count'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
    
    # =========================================================================
    # COINGECKO
    # =========================================================================
    
    def fetch_coingecko(self, coin_id: str = "bitcoin", days: int = 1) -> pd.DataFrame:
        """
        Fetch from CoinGecko REST API.
        
        Note: Free tier has limited granularity
        - 1-2 days: 5-minute intervals
        - 3-90 days: hourly intervals
        """
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        response = requests.get(url, params={
            'vs_currency': 'usd',
            'days': days,
        })
        
        if response.status_code == 429:
            raise Exception("CoinGecko rate limit reached. Wait 1 minute.")
        
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = 0.0  # CoinGecko OHLC doesn't include volume
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # =========================================================================
    # COINCAP
    # =========================================================================
    
    def fetch_coincap(self, asset_id: str = "bitcoin", limit: int = 1000) -> pd.DataFrame:
        """Fetch from CoinCap REST API (completely free)."""
        url = f"https://api.coincap.io/v2/assets/{asset_id}/history"
        response = requests.get(url, params={
            'interval': 'm1',  # 1 minute
        })
        response.raise_for_status()
        
        data = response.json()['data']
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['open'] = df['priceUsd'].astype(float)
        df['high'] = df['priceUsd'].astype(float)
        df['low'] = df['priceUsd'].astype(float)
        df['close'] = df['priceUsd'].astype(float)
        df['volume'] = 0.0  # CoinCap history doesn't have volume
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_source(self, name: str, df: pd.DataFrame) -> Dict:
        """
        Validate a data source.
        
        Returns:
            Validation report for this source
        """
        print(f"\n{'─' * 50}")
        print(f"Validating: {name}")
        print(f"{'─' * 50}")
        print(f"  Rows: {len(df)}")
        print(f"  First: {df['timestamp'].iloc[0]}")
        print(f"  Last:  {df['timestamp'].iloc[-1]}")
        print(f"  Price: ${df['close'].iloc[-1]:,.2f}")
        
        # Run Level 1
        level1 = self.validator.validate_level1(df)
        
        # Run Level 2 if Level 1 passed
        if level1.passed and len(df) >= 100:
            level2 = self.validator.validate_level2(df)
        else:
            from validation.ohlcv_validator import ValidationResult
            level2 = ValidationResult(level=2, passed=False, 
                                       issues=["Skipped (L1 failed or insufficient data)"])
        
        result = {
            'name': name,
            'rows': len(df),
            'level1_passed': level1.passed,
            'level2_passed': level2.passed,
            'overall_passed': level1.passed and level2.passed,
            'level1_issues': level1.issues,
            'level2_issues': level2.issues,
            'level1_warnings': level1.warnings,
            'level2_warnings': level2.warnings,
        }
        
        status = "✓ PASSED" if result['overall_passed'] else "✗ FAILED"
        print(f"  Result: {status}")
        
        if level1.issues:
            for issue in level1.issues:
                print(f"    ✗ L1: {issue}")
        if level2.issues and level2.issues != ["Skipped (L1 failed or insufficient data)"]:
            for issue in level2.issues:
                print(f"    ✗ L2: {issue}")
        
        self.results[name] = result
        return result
    
    def validate_all(self) -> Dict:
        """Validate all data sources."""
        print("=" * 60)
        print("HIMARI DATA SOURCE VALIDATION")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Binance (Primary)
        try:
            print("\n[1/4] Fetching Binance data...")
            df_binance = self.fetch_binance("BTCUSDT", 1000)
            self.validate_source("Binance (BTCUSDT)", df_binance)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.results["Binance"] = {'error': str(e), 'overall_passed': False}
        
        # Kraken (Backup)
        try:
            print("\n[2/4] Fetching Kraken data...")
            df_kraken = self.fetch_kraken("XXBTZUSD")
            self.validate_source("Kraken (XXBTZUSD)", df_kraken)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.results["Kraken"] = {'error': str(e), 'overall_passed': False}
        
        # CoinGecko (Supplemental)
        try:
            print("\n[3/4] Fetching CoinGecko data...")
            df_coingecko = self.fetch_coingecko("bitcoin", 1)
            self.validate_source("CoinGecko (bitcoin)", df_coingecko)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.results["CoinGecko"] = {'error': str(e), 'overall_passed': False}
        
        # CoinCap (Free unlimited)
        try:
            print("\n[4/4] Fetching CoinCap data...")
            df_coincap = self.fetch_coincap("bitcoin", 1000)
            self.validate_source("CoinCap (bitcoin)", df_coincap)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.results["CoinCap"] = {'error': str(e), 'overall_passed': False}
        
        return self.results


def main():
    """Run validation on all sources."""
    validator = DataSourceValidator()
    results = validator.validate_all()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        if result.get('overall_passed'):
            print(f"  ✓ {name}: VALID")
            passed += 1
        else:
            error = result.get('error', 'Validation failed')
            print(f"  ✗ {name}: {error}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Recommendation
    if passed > 0:
        print("\n📊 RECOMMENDATION:")
        if 'Binance' in results and results.get('Binance (BTCUSDT)', {}).get('overall_passed'):
            print("  → Use BINANCE as primary source (lowest latency, highest quality)")
        if 'Kraken' in results and results.get('Kraken (XXBTZUSD)', {}).get('overall_passed'):
            print("  → Use KRAKEN as backup source")
        print()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
