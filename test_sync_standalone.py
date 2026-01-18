"""
Standalone synchronous interface test for TimescaleDB Backtester.
Run this separately from the async tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.validation.timescale_backtester import create_timescale_backtester
import numpy as np


def main():
    """Test synchronous interface for HIFA compatibility."""
    print("="*60)
    print("SYNC INTERFACE TEST (Standalone)")
    print("="*60)

    backtester = create_timescale_backtester()

    class MockStrategy:
        def __init__(self):
            self.id = "sync_test_strategy"
            self.root = None

        def to_vector(self):
            weights = np.zeros(127)
            weights[0] = 0.5
            return weights

    strategy = MockStrategy()

    try:
        # Test quick_eval (DSR gate)
        print("\n[TEST] quick_eval()...")
        quick_result = backtester.quick_eval(strategy)
        print(f"  Result: {quick_result}")
        print("  [PASS]")

        # Test run (fast backtest)
        print("\n[TEST] run()...")
        result = backtester.run(
            strategy=strategy,
            assets="top20",
            start_date="2023-06-01",
            end_date="2023-12-01",
            execution_model="instant"
        )
        print(f"  Sharpe: {result.sharpe:.2f}")
        print(f"  Max DD: {result.max_drawdown:.2%}")
        print(f"  Trades: {result.trade_count}")
        print("  [PASS]")

        # Test get_returns (CPCV)
        print("\n[TEST] get_returns()...")
        returns = backtester.get_returns(strategy)
        print(f"  Shape: {returns.shape}")
        print(f"  Non-zero: {(returns != 0).sum()}")
        print("  [PASS]")

        print("\n" + "="*60)
        print("ALL SYNC TESTS PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
