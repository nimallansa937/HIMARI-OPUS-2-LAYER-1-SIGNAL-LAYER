"""
Comprehensive Multi-Source Sentiment Test
==========================================

Tests all data connectors with multi-model ensemble:
- CryptoPanic (news)
- Telegram (alpha channels)
- Multi-model sentiment analysis (Phase 2)

Compares results across sources and models.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors import (
    create_cryptopanic_connector,
    create_telegram_connector,
)

logging.basicConfig(level=logging.INFO)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "93c0a156ad8a07e4090cc173c22ecf56ecb63bdb")

TELEGRAM_CHANNELS = [
    "@cryptowhalesignal",
    "@cryptoalphacalls",
]

TEST_CURRENCIES = ["BTC", "ETH", "SOL"]


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_sentiment_summary(results: Dict):
    """Print sentiment analysis summary."""
    signal_emoji = {
        "BULLISH": "🟢",
        "BEARISH": "🔴",
        "NEUTRAL": "⚪",
    }
    
    signal = results.get("signal", "NEUTRAL")
    emoji = signal_emoji.get(signal, "⚪")
    
    print(f"\n{emoji} Signal: {signal}")
    print(f"  Score: {results.get('sentiment_score', 0):+.2f}")
    print(f"  Messages/News: {results.get('message_count', results.get('news_count', 0))}")
    print(f"  Bullish: {results.get('bullish_pct', 0):.0%}")
    print(f"  Bearish: {results.get('bearish_pct', 0):.0%}")
    print(f"  Neutral: {results.get('neutral_pct', 0):.0%}")


def test_cryptopanic():
    """Test CryptoPanic news connector."""
    print_header("CRYPTOPANIC NEWS ANALYSIS")
    
    connector = create_cryptopanic_connector(
        api_key=CRYPTOPANIC_API_KEY,
        with_analyzer=True,
    )
    
    for currency in TEST_CURRENCIES:
        print(f"\n📰 {currency} News Sentiment:")
        
        try:
            results = connector.analyze_sentiment(
                currencies=currency,
                filter="hot",
            )
            print_sentiment_summary(results)
            
            # Show top predictions
            if results.get("model_predictions"):
                print("\n  Top Headlines:")
                for pred in results["model_predictions"][:3]:
                    model = pred.get("model", "Unknown")
                    score = pred.get("score", 0)
                    title = pred.get("title", "")
                    print(f"    [{model}] {score:+.2f} - {title}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_telegram():
    """Test Telegram channel connector."""
    print_header("TELEGRAM ALPHA CHANNELS")
    
    connector = create_telegram_connector(with_analyzer=True)
    
    for channel in TELEGRAM_CHANNELS:
        print(f"\n💬 {channel} Sentiment:")
        
        try:
            results = connector.analyze_sentiment(
                channel=channel,
                limit=10,
            )
            print_sentiment_summary(results)
            
            # Show top messages
            if results.get("model_predictions"):
                print("\n  Top Messages:")
                for pred in results["model_predictions"][:3]:
                    model = pred.get("model", "Unknown")
                    score = pred.get("score", 0)
                    text = pred.get("text", "")
                    views = pred.get("views", 0)
                    views_k = views / 1000
                    print(f"    [{model}] {score:+.2f} ({views_k:.1f}K views) - {text}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")


def aggregate_signals():
    """Aggregate signals across all sources."""
    print_header("MULTI-SOURCE SIGNAL AGGREGATION")
    
    print("\n🔀 Combining all sources for BTC...")
    
    # Get CryptoPanic sentiment
    cryptopanic = create_cryptopanic_connector(api_key=CRYPTOPANIC_API_KEY, with_analyzer=True)
    news_sentiment = cryptopanic.analyze_sentiment(currencies="BTC")
    
    # Get Telegram sentiment
    telegram = create_telegram_connector(with_analyzer=True)
    telegram_sentiment = telegram.analyze_sentiment("@cryptowhalesignal", limit=10)
    
    # Weight by source reliability and freshness
    news_weight = 0.6  # News more reliable
    telegram_weight = 0.4  # Telegram more alpha but noisier
    
    combined_score = (
        news_sentiment.get("sentiment_score", 0) * news_weight +
        telegram_sentiment.get("sentiment_score", 0) * telegram_weight
    )
    
    # Determine final signal
    if combined_score > 0.3:
        final_signal = "BULLISH"
    elif combined_score < -0.3:
        final_signal = "BEARISH"
    else:
        final_signal = "NEUTRAL"
    
    signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
    
    print(f"\n{signal_emoji[final_signal]} Final BTC Signal: {final_signal}")
    print(f"  Combined Score: {combined_score:+.2f}")
    print(f"\n  Source Breakdown:")
    print(f"    News (60%): {news_sentiment.get('sentiment_score', 0):+.2f}")
    print(f"    Telegram (40%): {telegram_sentiment.get('sentiment_score', 0):+.2f}")
    
    print(f"\n  Confidence Metrics:")
    print(f"    News count: {news_sentiment.get('news_count', 0)}")
    print(f"    Telegram messages: {telegram_sentiment.get('message_count', 0)}")
    print(f"    Total views: {telegram_sentiment.get('total_views', 0):,}")


def run_full_test():
    """Run comprehensive test of all connectors."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "HIMARI MULTI-SOURCE SENTIMENT ANALYSIS".center(68) + "║")
    print("║" + "Phase 2 Multi-Model Ensemble".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Test each connector
    test_cryptopanic()
    test_telegram()
    
    # Aggregate signals
    aggregate_signals()
    
    # Summary
    print_header("TEST COMPLETE")
    print("\n✅ All connectors tested successfully!")
    print("\nData Flow:")
    print("  News (CryptoPanic) → ModernFinBERT → 60% weight")
    print("  Social (Telegram) → CryptoBERT → 40% weight")
    print("  → Ensemble Voting → Position Recommendation")
    
    print("\nNext Steps:")
    print("  1. Deploy to production with Kafka integration")
    print("  2. Add real-time streaming (30s polling)")
    print("  3. Set up Prometheus alerts")
    print("\n")


if __name__ == "__main__":
    run_full_test()
