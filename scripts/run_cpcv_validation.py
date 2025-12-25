"""
CPCV Validation Runner - Historical Backtesting

Runs Combinatorial Purged Cross-Validation on historical data
to validate the Enhanced Layer 1 signal improvements.

Expected improvements:
- Sharpe Ratio: 0.8-1.2 → 1.3-1.9 (+0.4 to +0.6)
- Win Rate: Maintain > 45%
- Max Drawdown: < 25%
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from validation.cpcv_validator import CPCVValidator, CPCVConfig
from primitives import IntegratedSignalLayer
from config import load_enhanced_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_data(symbol='BTCUSDT', start_date='2020-01-01', end_date='2024-12-31'):
    """
    Load historical OHLCV data for validation.
    
    In production, this would load from your data warehouse.
    For now, we'll generate simulated data matching crypto characteristics.
    """
    logger.info(f"Loading historical data: {symbol} from {start_date} to {end_date}")
    
    # Generate simulated data (replace with real data loading)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    n = len(dates)
    
    # Simulate realistic crypto price action
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n)  # Slight positive drift, 2% hourly vol
    
    # Add regime changes
    for i in range(0, n, 500):
        end_idx = min(i + 500, n)
        actual_len = end_idx - i
        if i % 1000 == 0:
            # Bull regime
            returns[i:end_idx] = np.random.normal(0.002, 0.015, actual_len)
        else:
            # Bear/Range regime
            returns[i:end_idx] = np.random.normal(-0.001, 0.025, actual_len)
    
    prices = 10000 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n)
    })
    
    logger.info(f"Loaded {len(df)} bars from {df.timestamp.min()} to {df.timestamp.max()}")
    logger.info(f"Price range: ${df.close.min():.2f} - ${df.close.max():.2f}")
    
    return df


def baseline_strategy(train_df, test_df):
    """
    Baseline strategy: Simple RSI mean reversion.
    
    This represents the "before" performance without Enhanced Layer 1.
    """
    test_signals = []
    test_returns = []
    
    # Simple RSI calculation
    delta = test_df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    for i in range(len(test_df)):
        if i < 14:
            test_signals.append(0)
            test_returns.append(0)
        else:
            # RSI mean reversion
            if rsi.iloc[i] < 30:
                signal = 1.0  # Oversold, buy
            elif rsi.iloc[i] > 70:
                signal = -1.0  # Overbought, sell
            else:
                signal = 0.0
            
            test_signals.append(signal)
            
            # Calculate return
            if i < len(test_df) - 1:
                ret = (test_df['close'].iloc[i+1] - test_df['close'].iloc[i]) / test_df['close'].iloc[i]
                test_returns.append(signal * ret)
            else:
                test_returns.append(0)
    
    return test_signals, np.array(test_returns)


def enhanced_strategy(train_df, test_df):
    """
    Enhanced strategy using IntegratedSignalLayer.
    
    This represents the "after" performance with all 7 primitives.
    """
    config = load_enhanced_config()
    layer = IntegratedSignalLayer(config, redis_client=None)
    
    test_signals = []
    test_returns = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        ohlcv = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }
        
        # Generate signal
        output = layer.update('BTCUSDT', ohlcv)
        test_signals.append(output.composite_signal)
        
        # Calculate return
        if i < len(test_df) - 1:
            ret = (test_df['close'].iloc[i+1] - test_df['close'].iloc[i]) / test_df['close'].iloc[i]
            test_returns.append(output.composite_signal * ret)
        else:
            test_returns.append(0)
    
    return test_signals, np.array(test_returns)


def run_cpcv_validation():
    """Run complete CPCV validation comparing baseline vs enhanced."""
    
    print("=" * 70)
    print("CPCV VALIDATION - Enhanced Layer 1 Signal System")
    print("=" * 70)
    print()
    
    # Load historical data
    df = load_historical_data('BTCUSDT', '2020-01-01', '2024-12-31')
    
    # Initialize validator
    config = CPCVConfig(
        n_splits=5,
        purge_bars=10,
        embargo_bars=5,
        min_sharpe=0.8,
        min_win_rate=0.45,
        max_drawdown=0.25,
        n_strategies_tested=2  # Baseline + Enhanced
    )
    
    validator = CPCVValidator(config)
    
    # Test baseline strategy
    print("Testing BASELINE strategy (Simple RSI)...")
    print("-" * 70)
    baseline_results = validator.validate(df, baseline_strategy, price_col='close')
    
    print("\nBaseline Results:")
    print(validator.get_validation_report(baseline_results))
    print()

    # Test enhanced strategy
    print("Testing ENHANCED strategy (IntegratedSignalLayer)...")
    print("-" * 70)
    enhanced_results = validator.validate(df, enhanced_strategy, price_col='close')

    print("\nEnhanced Results:")
    print(validator.get_validation_report(enhanced_results))
    print()
    
    # Compare results
    print("=" * 70)
    print("COMPARISON - Baseline vs Enhanced")
    print("=" * 70)
    
    baseline_sharpe = baseline_results.get('sharpe', 0)
    enhanced_sharpe = enhanced_results.get('sharpe', 0)
    sharpe_improvement = enhanced_sharpe - baseline_sharpe
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<20} {baseline_sharpe:<15.2f} {enhanced_sharpe:<15.2f} {sharpe_improvement:+<15.2f}")
    print(f"{'Win Rate':<20} {baseline_results.get('win_rate', 0):<15.1%} {enhanced_results.get('win_rate', 0):<15.1%}")
    print(f"{'Max Drawdown':<20} {baseline_results.get('max_drawdown', 0):<15.1%} {enhanced_results.get('max_drawdown', 0):<15.1%}")
    print(f"{'Total Return':<20} {baseline_results.get('total_return', 0):<15.1%} {enhanced_results.get('total_return', 0):<15.1%}")
    print()
    
    # Check if improvement meets target
    target_sharpe_improvement = 0.4
    
    if sharpe_improvement >= target_sharpe_improvement:
        print(f"SUCCESS: Sharpe improved by {sharpe_improvement:+.2f} (target: +{target_sharpe_improvement})")
    else:
        print(f"WARNING: Sharpe improvement {sharpe_improvement:+.2f} below target +{target_sharpe_improvement}")
    
    print()
    print("=" * 70)
    print("CPCV Validation Complete")
    print("=" * 70)
    
    return {
        'baseline': baseline_results,
        'enhanced': enhanced_results,
        'sharpe_improvement': sharpe_improvement
    }


if __name__ == '__main__':
    results = run_cpcv_validation()
