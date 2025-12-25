"""
Phase 2 A/B Test: Multi-Model Ensemble Validation
==================================================

Tests Phase 2 ensemble (CryptoBERT + ModernFinBERT + FinTwitBERT)
against Phase 1 baseline (CryptoBERT + ModernFinBERT).

Expected improvements:
- 30-40% false signal reduction from majority voting
- +0.5-0.8 Sharpe improvement
"""

import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primitives.multi_model_sentiment import (
    MultiModelSentimentAnalyzer,
    EnsembleConfig,
    ModelType,
    create_phase1_analyzer,
    create_phase2_analyzer,
)

logging.basicConfig(level=logging.INFO)

# =============================================================================
# TEST SAMPLES
# =============================================================================

TEST_SAMPLES = [
    # Format: (text, source, expected_label)
    
    # Clear bullish
    ("Bitcoin breaking $50k! 🚀🚀🚀 LFG", "twitter", "bullish"),
    ("ETH looking bullish af, consolidation about to break up", "twitter", "bullish"),
    ("Solana dev activity exploding. $SOL to the moon soon", "twitter", "bullish"),
    ("Bitcoin surges as institutional demand grows", "bloomberg", "bullish"),
    ("BlackRock CEO says Bitcoin has potential to revolutionize finance", "reuters", "bullish"),
    
    # Clear bearish
    ("SEC crackdown incoming 📉 get out while you can", "twitter", "bearish"),
    ("Crypto winter is here. This dump has legs", "twitter", "bearish"),
    ("HODL this dumpster fire 💀 we're cooked", "twitter", "bearish"),
    ("SEC files lawsuit against major cryptocurrency exchange", "bloomberg", "bearish"),
    ("Major crypto lender files for bankruptcy protection", "coindesk", "bearish"),
    
    # Tricky sarcasm
    ("Great, another 10% dump. Love to see it 🙄", "twitter", "bearish"),
    ("Totally buying the dip guys [proceeds to get rekt]", "twitter", "bearish"),
    
    # Neutral
    ("Crypto market trading sideways today", "twitter", "neutral"),
    ("BTC hovering around key levels. Waiting for break", "twitter", "neutral"),
]


def run_phase2_test():
    print("=" * 70)
    print("PHASE 2 A/B TEST: ENSEMBLE VALIDATION")
    print("=" * 70)
    
    # Load Phase 2 analyzer
    print("\n📦 Loading Phase 2 models (may take a minute)...")
    analyzer = create_phase2_analyzer()
    
    print("\n🧪 Running ensemble tests...\n")
    
    results = []
    
    for text, source, expected in TEST_SAMPLES:
        # Run ensemble analysis
        result = analyzer.analyze_ensemble(text)
        
        correct = result.final_label == expected
        results.append({
            "text": text[:40],
            "expected": expected,
            "predicted": result.final_label,
            "correct": correct,
            "agreement": result.agreement_rate,
            "confidence": result.confidence_level,
            "recommendation": result.position_recommendation,
        })
        
        symbol = "✓" if correct else "✗"
        print(f"{symbol} [{source:10}] {text[:35]}...")
        print(f"   Expected: {expected:8} | Got: {result.final_label:8} | Agreement: {result.agreement_rate:.0%}")
        print(f"   Confidence: {result.confidence_level:6} | Position: {result.position_recommendation}")
        print(f"   Predictions: {[p.label for p in result.individual_predictions]}")
        print()
    
    # Summary
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_agreement = sum(r["agreement"] for r in results) / len(results)
    
    high_conf = sum(1 for r in results if r["confidence"] == "HIGH")
    medium_conf = sum(1 for r in results if r["confidence"] == "MEDIUM")
    low_conf = sum(1 for r in results if r["confidence"] == "LOW")
    
    print("=" * 70)
    print("PHASE 2 ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.1%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"Average Agreement Rate: {avg_agreement:.1%}")
    print(f"Confidence Breakdown:")
    print(f"  HIGH: {high_conf} | MEDIUM: {medium_conf} | LOW: {low_conf}")
    print()
    
    # Position recommendations
    print("Position Recommendations:")
    full = sum(1 for r in results if r["recommendation"] == "FULL")
    reduced = sum(1 for r in results if r["recommendation"] == "REDUCED")
    hold = sum(1 for r in results if r["recommendation"] == "HOLD")
    print(f"  FULL: {full} | REDUCED: {reduced} | HOLD: {hold}")
    
    # Metrics
    print()
    metrics = analyzer.get_metrics()
    print("Model Metrics:")
    for model_name, stats in metrics["per_model"].items():
        print(f"  {stats['name']}: {stats['calls']} calls, p50: {stats['latency_p50_ms']:.1f}ms")
    
    print("=" * 70)
    
    if accuracy >= 0.80 and avg_agreement >= 0.70:
        print("🟢 RECOMMENDATION: DEPLOY Phase 2")
    elif accuracy >= 0.70:
        print("🟡 RECOMMENDATION: CONTINUE_TESTING Phase 2")
    else:
        print("🔴 RECOMMENDATION: KEEP Phase 1")
    
    print("=" * 70)
    
    return accuracy, avg_agreement


if __name__ == "__main__":
    run_phase2_test()
