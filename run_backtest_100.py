"""
Generate 100 random strategies and backtest them through HIFA pipeline.

This script:
1. Generates 100 random strategy genomes
2. Runs each through TimescaleDB-backed backtester
3. Reports results and statistics
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.genome import generate_random_strategy, StrategyGenome
from src.core.grammar import GrammarValidator
from src.validation.timescale_backtester import create_timescale_backtester
import numpy as np


def main():
    print("="*70)
    print("LAYER 1 EXPLORER: Generate & Backtest 100 Random Strategies")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize components
    print("[1/4] Initializing backtester...")
    backtester = create_timescale_backtester()
    grammar = GrammarValidator()

    # Generate strategies
    print("[2/4] Generating 100 random strategies...")
    strategies = []
    for i in range(100):
        strategy = generate_random_strategy(max_depth=4)
        strategies.append(strategy)
    print(f"      Generated {len(strategies)} strategies")

    # Validate grammar
    print("[3/4] Validating grammar...")
    valid_strategies = []
    for s in strategies:
        is_valid, errors = grammar.validate_genome(s)
        if is_valid:
            valid_strategies.append(s)
    print(f"      {len(valid_strategies)}/{len(strategies)} passed grammar validation")

    # Backtest
    print("[4/4] Running backtests on TimescaleDB historical data...")
    print("      (5.4M feature vectors, 2020-2024, 13 symbols)")
    print()

    results = []
    start_time = time.time()

    for i, strategy in enumerate(valid_strategies):
        try:
            # Quick backtest (1 year, single symbol for speed)
            result = backtester.run(
                strategy=strategy,
                assets="BTCUSDT",
                start_date="2023-01-01",
                end_date="2024-01-01",
                execution_model="realistic"
            )

            results.append({
                'id': strategy.id,
                'sharpe': result.sharpe,
                'max_dd': result.max_drawdown,
                'trades': result.trade_count,
                'profit_factor': result.profit_factor,
                'total_return': result.total_return,
                'win_rate': result.win_rate
            })

            # Progress indicator
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i + 1)) * (len(valid_strategies) - i - 1)
                print(f"      Progress: {i+1}/{len(valid_strategies)} "
                      f"(~{remaining:.0f}s remaining)")

        except Exception as e:
            print(f"      [WARN] Strategy {strategy.id}: {e}")
            continue

    elapsed_total = time.time() - start_time

    # Statistics
    print()
    print("="*70)
    print("BACKTEST RESULTS SUMMARY")
    print("="*70)

    if not results:
        print("No strategies completed backtesting!")
        return

    sharpes = [r['sharpe'] for r in results]
    max_dds = [r['max_dd'] for r in results]
    profit_factors = [r['profit_factor'] for r in results]

    print(f"\nTotal strategies tested: {len(results)}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/len(results):.2f}s per strategy)")
    print()

    print("Sharpe Ratio Distribution:")
    print(f"  Min:    {min(sharpes):.2f}")
    print(f"  Max:    {max(sharpes):.2f}")
    print(f"  Mean:   {np.mean(sharpes):.2f}")
    print(f"  Median: {np.median(sharpes):.2f}")
    print(f"  Std:    {np.std(sharpes):.2f}")
    print()

    print("Max Drawdown Distribution:")
    print(f"  Min:    {min(max_dds):.2%}")
    print(f"  Max:    {max(max_dds):.2%}")
    print(f"  Mean:   {np.mean(max_dds):.2%}")
    print()

    print("Profit Factor Distribution:")
    print(f"  Min:    {min(profit_factors):.2f}")
    print(f"  Max:    {max(profit_factors):.2f}")
    print(f"  Mean:   {np.mean(profit_factors):.2f}")
    print()

    # Filter promising strategies
    promising = [r for r in results if r['sharpe'] > 0 and r['max_dd'] < 0.5]
    print(f"Promising strategies (Sharpe > 0, DD < 50%): {len(promising)}")

    very_promising = [r for r in results if r['sharpe'] > 1.0 and r['max_dd'] < 0.3]
    print(f"Very promising (Sharpe > 1.0, DD < 30%): {len(very_promising)}")

    # Show top 5
    print()
    print("="*70)
    print("TOP 5 STRATEGIES BY SHARPE RATIO")
    print("="*70)

    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for i, r in enumerate(sorted_results, 1):
        print(f"\n#{i}: {r['id'][:16]}...")
        print(f"    Sharpe:        {r['sharpe']:.2f}")
        print(f"    Max Drawdown:  {r['max_dd']:.2%}")
        print(f"    Profit Factor: {r['profit_factor']:.2f}")
        print(f"    Total Return:  {r['total_return']:.2%}")
        print(f"    Win Rate:      {r['win_rate']:.2%}")
        print(f"    Trade Count:   {r['trades']}")

    print()
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
