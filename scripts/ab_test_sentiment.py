"""
A/B Testing Framework for HIMARI Sentiment Models
==================================================

Compares the new multi-model ensemble (Phase 1: CryptoBERT + ModernFinBERT)
against the baseline dual-path analyzer (Twitter-RoBERTa + Financial-RoBERTa).

Metrics tracked:
- Accuracy (directional correctness vs price movement)
- Latency (p50, p95, p99)
- Confidence calibration
- Signal quality (Sharpe improvement estimate)

Usage:
    python scripts/ab_test_sentiment.py --samples 1000 --output results/
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# SAMPLE DATA
# =============================================================================

# Labeled test samples with expected sentiment and actual price outcome
CRYPTO_TEST_SAMPLES = [
    # Format: (text, source, expected_label, price_moved_correctly)
    
    # Bullish tweets - should predict positive, market should move up
    ("Bitcoin breaking $50k! 🚀🚀🚀 LFG", "twitter", "bullish", True),
    ("ETH looking bullish af, consolidation about to break up", "twitter", "bullish", True),
    ("Accumulating more BTC here. This is the dip before the rip 🔥", "twitter", "bullish", True),
    ("Solana dev activity exploding. $SOL to the moon soon", "twitter", "bullish", True),
    ("Diamond hands paying off. Never selling 💎🙌", "twitter", "bullish", True),
    ("Just bought more. The fear is the opportunity", "twitter", "bullish", True),
    ("Bullish divergence on the 4h. Loading up", "reddit", "bullish", True),
    ("Institutional buying incoming. Charts don't lie", "telegram", "bullish", True),
    
    # Bearish tweets - should predict negative, market should move down
    ("SEC crackdown incoming 📉 get out while you can", "twitter", "bearish", True),
    ("Crypto winter is here. This dump has legs", "twitter", "bearish", True),
    ("HODL this dumpster fire 💀 we're cooked", "twitter", "bearish", True),
    ("Massive sell wall at 48k. Bears in control", "twitter", "bearish", True),
    ("Whale dumping detected. Panic sell time", "telegram", "bearish", True),
    ("Exchange outflows crashing. Nobody wants this trash", "reddit", "bearish", True),
    ("Death cross forming. Prepare for 30k", "twitter", "bearish", True),
    ("Funding rates negative. Smart money shorting", "twitter", "bearish", True),
    
    # Sarcasm/irony tests - tricky cases
    ("Great, another 10% dump. Love to see it 🙄", "twitter", "bearish", True),
    ("Totally buying the dip guys [proceeds to get rekt]", "twitter", "bearish", True),
    ("Number go up technology working perfectly 📉📉📉", "twitter", "bearish", True),
    
    # Neutral tweets
    ("Crypto market trading sideways today", "twitter", "neutral", True),
    ("Volume looking average. No clear direction", "twitter", "neutral", True),
    ("BTC hovering around key levels. Waiting for break", "twitter", "neutral", True),
    
    # Bullish news - high confidence trades
    ("Bitcoin surges past $50,000 as institutional demand grows", "bloomberg", "bullish", True),
    ("BlackRock CEO says Bitcoin has potential to revolutionize finance", "reuters", "bullish", True),
    ("Spot Bitcoin ETF approval expected within weeks, sources say", "coindesk", "bullish", True),
    ("MicroStrategy announces additional $500M Bitcoin purchase", "bloomberg", "bullish", True),
    ("Fidelity launches crypto trading for retail investors", "reuters", "bullish", True),
    ("El Salvador reports positive returns on Bitcoin treasury", "news", "bullish", True),
    
    # Bearish news - high confidence trades
    ("SEC files lawsuit against major cryptocurrency exchange", "bloomberg", "bearish", True),
    ("China reiterates crypto trading ban with stricter enforcement", "reuters", "bearish", True),
    ("Major crypto lender files for bankruptcy protection", "coindesk", "bearish", True),
    ("FBI warns of North Korean hackers targeting crypto firms", "bloomberg", "bearish", True),
    ("Crypto market faces regulatory headwinds in multiple jurisdictions", "reuters", "bearish", True),
    ("Mt. Gox creditors to receive $9B in Bitcoin distributions", "news", "bearish", True),
    
    # False signals (for testing - expected label doesn't match outcome)
    ("BTC going to the moon! 🚀", "twitter", "bullish", False),  # Pump and dump
    ("Perfect buying opportunity here", "twitter", "bullish", False),  # Caught falling knife
    ("Massive crash incoming", "twitter", "bearish", False),  # Bear trap
]


@dataclass
class ABTestResult:
    """Result from A/B test comparison."""
    sample_text: str
    source: str
    expected_label: str
    price_correct: bool
    
    # Model A (baseline) results
    model_a_score: float
    model_a_label: str
    model_a_confidence: float
    model_a_latency_ms: float
    model_a_correct: bool
    
    # Model B (new) results
    model_b_score: float
    model_b_label: str
    model_b_confidence: float
    model_b_latency_ms: float
    model_b_correct: bool
    
    # Comparison
    agreement: bool
    better_model: str  # "A", "B", "tie"


@dataclass
class ABTestSummary:
    """Summary statistics from A/B test."""
    total_samples: int
    timestamp: str
    
    # Accuracy
    model_a_accuracy: float
    model_b_accuracy: float
    accuracy_improvement: float
    
    # Latency
    model_a_latency_p50: float
    model_a_latency_p95: float
    model_b_latency_p50: float
    model_b_latency_p95: float
    latency_improvement_pct: float
    
    # Agreement
    agreement_rate: float
    
    # Confidence calibration
    model_a_avg_confidence: float
    model_b_avg_confidence: float
    
    # Winners
    model_a_wins: int
    model_b_wins: int
    ties: int
    
    # Sharpe estimate (simplified)
    estimated_sharpe_improvement: float
    
    # Per-source breakdown
    per_source: Dict[str, Dict] = field(default_factory=dict)
    
    # Recommendation
    recommendation: str = ""
    recommendation_confidence: str = ""


class SentimentABTest:
    """
    A/B Testing harness for sentiment models.
    
    Compares:
    - Model A (baseline): DualPathSentimentAnalyzer
    - Model B (new): MultiModelSentimentAnalyzer (Phase 1)
    
    Example:
        ab_test = SentimentABTest()
        summary = ab_test.run_comparison(samples=CRYPTO_TEST_SAMPLES)
        ab_test.save_results("results/ab_test_2025_12_25.json")
    """
    
    def __init__(
        self,
        model_a_name: str = "DualPath (Baseline)",
        model_b_name: str = "MultiModel Phase 1 (New)",
    ):
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        
        self._model_a = None
        self._model_b = None
        
        self._results: List[ABTestResult] = []
        self._summary: Optional[ABTestSummary] = None
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load both models for comparison."""
        try:
            # Model A: Baseline dual-path
            from primitives.dual_path_sentiment import create_dual_path_analyzer
            self._model_a = create_dual_path_analyzer(use_fine_tuned=False)
            logger.info(f"✓ Model A loaded: {self.model_a_name}")
        except Exception as e:
            logger.error(f"Failed to load Model A: {e}")
        
        try:
            # Model B: New multi-model
            from primitives.multi_model_sentiment import create_phase1_analyzer
            self._model_b = create_phase1_analyzer()
            logger.info(f"✓ Model B loaded: {self.model_b_name}")
        except Exception as e:
            logger.error(f"Failed to load Model B: {e}")
    
    def run_comparison(
        self,
        samples: Optional[List[Tuple]] = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> ABTestSummary:
        """
        Run A/B comparison on sample set.
        
        Args:
            samples: List of (text, source, expected_label, price_correct) tuples
            shuffle: Whether to randomize sample order
            verbose: Print progress
            
        Returns:
            ABTestSummary with comparison results
        """
        samples = samples or CRYPTO_TEST_SAMPLES
        
        if shuffle:
            samples = list(samples)
            random.shuffle(samples)
        
        self._results = []
        
        if verbose:
            print(f"\n🧪 Running A/B Test: {len(samples)} samples")
            print(f"   Model A: {self.model_a_name}")
            print(f"   Model B: {self.model_b_name}")
            print("-" * 60)
        
        for i, (text, source, expected_label, price_correct) in enumerate(samples):
            result = self._test_single(text, source, expected_label, price_correct)
            self._results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(samples)} samples...")
        
        # Compute summary
        self._summary = self._compute_summary()
        
        if verbose:
            self._print_summary()
        
        return self._summary
    
    def _test_single(
        self,
        text: str,
        source: str,
        expected_label: str,
        price_correct: bool,
    ) -> ABTestResult:
        """Test a single sample with both models."""
        
        # Model A prediction
        a_score, a_label, a_conf, a_latency = 0.0, "neutral", 0.0, 0.0
        if self._model_a:
            try:
                start = time.perf_counter()
                result_a = self._model_a.analyze(text, source)
                a_latency = (time.perf_counter() - start) * 1000
                a_score = result_a.score
                a_label = result_a.label
                a_conf = result_a.confidence
            except Exception as e:
                logger.error(f"Model A error: {e}")
        
        # Model B prediction
        b_score, b_label, b_conf, b_latency = 0.0, "neutral", 0.0, 0.0
        if self._model_b:
            try:
                start = time.perf_counter()
                result_b = self._model_b.analyze(text, source)
                b_latency = (time.perf_counter() - start) * 1000
                if result_b:
                    b_score = result_b.score
                    b_label = result_b.label
                    b_conf = result_b.confidence
            except Exception as e:
                logger.error(f"Model B error: {e}")
        
        # Evaluate correctness
        # Correct = predicted label matches expected AND price moved in that direction
        a_correct = (a_label == expected_label) and price_correct
        b_correct = (b_label == expected_label) and price_correct
        
        # Determine winner
        if a_correct and not b_correct:
            better = "A"
        elif b_correct and not a_correct:
            better = "B"
        elif a_correct and b_correct:
            # Both correct - prefer higher confidence or lower latency
            if abs(a_conf - b_conf) > 0.1:
                better = "A" if a_conf > b_conf else "B"
            elif abs(a_latency - b_latency) > 10:
                better = "A" if a_latency < b_latency else "B"
            else:
                better = "tie"
        else:
            better = "tie"
        
        return ABTestResult(
            sample_text=text[:50],
            source=source,
            expected_label=expected_label,
            price_correct=price_correct,
            model_a_score=a_score,
            model_a_label=a_label,
            model_a_confidence=a_conf,
            model_a_latency_ms=a_latency,
            model_a_correct=a_correct,
            model_b_score=b_score,
            model_b_label=b_label,
            model_b_confidence=b_conf,
            model_b_latency_ms=b_latency,
            model_b_correct=b_correct,
            agreement=a_label == b_label,
            better_model=better,
        )
    
    def _compute_summary(self) -> ABTestSummary:
        """Compute summary statistics from results."""
        if not self._results:
            return ABTestSummary(
                total_samples=0,
                timestamp=datetime.now().isoformat(),
                model_a_accuracy=0.0,
                model_b_accuracy=0.0,
                accuracy_improvement=0.0,
                model_a_latency_p50=0.0,
                model_a_latency_p95=0.0,
                model_b_latency_p50=0.0,
                model_b_latency_p95=0.0,
                latency_improvement_pct=0.0,
                agreement_rate=0.0,
                model_a_avg_confidence=0.0,
                model_b_avg_confidence=0.0,
                model_a_wins=0,
                model_b_wins=0,
                ties=0,
                estimated_sharpe_improvement=0.0,
            )
        
        n = len(self._results)
        
        # Accuracy
        a_correct = sum(1 for r in self._results if r.model_a_correct)
        b_correct = sum(1 for r in self._results if r.model_b_correct)
        a_accuracy = a_correct / n
        b_accuracy = b_correct / n
        accuracy_improvement = (b_accuracy - a_accuracy) / max(a_accuracy, 0.001) * 100
        
        # Latency
        a_latencies = [r.model_a_latency_ms for r in self._results if r.model_a_latency_ms > 0]
        b_latencies = [r.model_b_latency_ms for r in self._results if r.model_b_latency_ms > 0]
        
        a_p50 = np.percentile(a_latencies, 50) if a_latencies else 0.0
        a_p95 = np.percentile(a_latencies, 95) if a_latencies else 0.0
        b_p50 = np.percentile(b_latencies, 50) if b_latencies else 0.0
        b_p95 = np.percentile(b_latencies, 95) if b_latencies else 0.0
        
        latency_improvement = (a_p50 - b_p50) / max(a_p50, 0.001) * 100
        
        # Agreement
        agreement_rate = sum(1 for r in self._results if r.agreement) / n
        
        # Confidence
        a_conf = np.mean([r.model_a_confidence for r in self._results])
        b_conf = np.mean([r.model_b_confidence for r in self._results])
        
        # Winners
        a_wins = sum(1 for r in self._results if r.better_model == "A")
        b_wins = sum(1 for r in self._results if r.better_model == "B")
        ties = sum(1 for r in self._results if r.better_model == "tie")
        
        # Per-source breakdown
        sources = set(r.source for r in self._results)
        per_source = {}
        for source in sources:
            source_results = [r for r in self._results if r.source == source]
            source_n = len(source_results)
            per_source[source] = {
                "samples": source_n,
                "model_a_accuracy": sum(1 for r in source_results if r.model_a_correct) / source_n,
                "model_b_accuracy": sum(1 for r in source_results if r.model_b_correct) / source_n,
            }
        
        # Sharpe estimate (simplified: accuracy improvement → Sharpe improvement)
        # Rough heuristic: 5% accuracy improvement ≈ +0.5 Sharpe
        estimated_sharpe = accuracy_improvement / 10 * 0.5
        
        # Recommendation
        if b_accuracy >= a_accuracy * 1.05:  # 5% improvement threshold
            recommendation = "DEPLOY"
            rec_confidence = "HIGH" if accuracy_improvement >= 10 else "MEDIUM"
        elif b_accuracy >= a_accuracy * 0.98:  # Within 2%
            recommendation = "CONTINUE_TESTING"
            rec_confidence = "MEDIUM"
        else:
            recommendation = "KEEP_BASELINE"
            rec_confidence = "HIGH"
        
        return ABTestSummary(
            total_samples=n,
            timestamp=datetime.now().isoformat(),
            model_a_accuracy=a_accuracy,
            model_b_accuracy=b_accuracy,
            accuracy_improvement=accuracy_improvement,
            model_a_latency_p50=a_p50,
            model_a_latency_p95=a_p95,
            model_b_latency_p50=b_p50,
            model_b_latency_p95=b_p95,
            latency_improvement_pct=latency_improvement,
            agreement_rate=agreement_rate,
            model_a_avg_confidence=a_conf,
            model_b_avg_confidence=b_conf,
            model_a_wins=a_wins,
            model_b_wins=b_wins,
            ties=ties,
            estimated_sharpe_improvement=estimated_sharpe,
            per_source=per_source,
            recommendation=recommendation,
            recommendation_confidence=rec_confidence,
        )
    
    def _print_summary(self) -> None:
        """Print formatted summary."""
        s = self._summary
        
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        print(f"Timestamp: {s.timestamp}")
        print(f"Samples: {s.total_samples}")
        
        print("\n📊 ACCURACY")
        print("-" * 40)
        print(f"Model A ({self.model_a_name}): {s.model_a_accuracy:.1%}")
        print(f"Model B ({self.model_b_name}): {s.model_b_accuracy:.1%}")
        improvement_symbol = "📈" if s.accuracy_improvement > 0 else "📉"
        print(f"Improvement: {improvement_symbol} {s.accuracy_improvement:+.1f}%")
        
        print("\n⚡ LATENCY")
        print("-" * 40)
        print(f"Model A p50: {s.model_a_latency_p50:.1f}ms | p95: {s.model_a_latency_p95:.1f}ms")
        print(f"Model B p50: {s.model_b_latency_p50:.1f}ms | p95: {s.model_b_latency_p95:.1f}ms")
        lat_symbol = "✓" if s.latency_improvement_pct > 0 else "✗"
        print(f"Improvement: {lat_symbol} {s.latency_improvement_pct:+.1f}%")
        
        print("\n🤝 AGREEMENT")
        print("-" * 40)
        print(f"Model agreement rate: {s.agreement_rate:.1%}")
        
        print("\n🏆 WINS")
        print("-" * 40)
        print(f"Model A wins: {s.model_a_wins}")
        print(f"Model B wins: {s.model_b_wins}")
        print(f"Ties: {s.ties}")
        
        print("\n📈 ESTIMATED SHARPE IMPROVEMENT")
        print("-" * 40)
        print(f"Estimated: {s.estimated_sharpe_improvement:+.2f}")
        
        print("\n📋 PER-SOURCE BREAKDOWN")
        print("-" * 40)
        for source, data in s.per_source.items():
            a_acc = data['model_a_accuracy']
            b_acc = data['model_b_accuracy']
            better = "✓ B" if b_acc > a_acc else ("= tie" if b_acc == a_acc else "✓ A")
            print(f"  {source:12} | A: {a_acc:.0%} | B: {b_acc:.0%} | {better}")
        
        print("\n" + "=" * 60)
        rec_color = {"DEPLOY": "🟢", "CONTINUE_TESTING": "🟡", "KEEP_BASELINE": "🔴"}
        print(f"RECOMMENDATION: {rec_color.get(s.recommendation, '')} {s.recommendation}")
        print(f"Confidence: {s.recommendation_confidence}")
        print("=" * 60)
    
    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "summary": asdict(self._summary) if self._summary else {},
            "results": [asdict(r) for r in self._results],
            "config": {
                "model_a_name": self.model_a_name,
                "model_b_name": self.model_b_name,
            }
        }
        
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output}")
    
    def get_detailed_results(self) -> List[ABTestResult]:
        """Get detailed per-sample results."""
        return self._results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A/B Test Sentiment Models")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--output", type=str, default="results/ab_test.json", help="Output path")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle samples")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("HIMARI SENTIMENT A/B TEST")
    print("=" * 60)
    
    # Run test
    ab_test = SentimentABTest()
    
    samples = CRYPTO_TEST_SAMPLES
    if args.samples:
        samples = samples[:args.samples]
    
    summary = ab_test.run_comparison(
        samples=samples,
        shuffle=args.shuffle,
        verbose=args.verbose,
    )
    
    # Save results
    ab_test.save_results(args.output)
    
    # Return exit code based on recommendation
    if summary.recommendation == "DEPLOY":
        return 0
    elif summary.recommendation == "CONTINUE_TESTING":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())
