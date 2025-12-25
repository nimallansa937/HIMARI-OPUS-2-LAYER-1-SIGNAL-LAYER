"""
Test Hybrid Sentiment Analyzer

Demonstrates VADER + FinBERT with crypto-specific lexicon.
"""

import sys
import io

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from primitives import HybridSentimentAnalyzer, HybridSentimentConfig

def test_crypto_sentiment():
    """Test sentiment on crypto-specific phrases."""

    print("=" * 60)
    print("Hybrid Sentiment Analyzer Test")
    print("=" * 60)
    print()

    # Initialize (will use VADER + attempt FinBERT)
    analyzer = HybridSentimentAnalyzer()

    if analyzer.vader is None:
        print("[!] VADER not available")
        return

    print("[+] VADER initialized with crypto lexicon")
    if analyzer.finbert is not None:
        print("[+] FinBERT initialized")
    else:
        print("[!] FinBERT not initialized (using VADER only)")
    print()

    # Test crypto-specific phrases
    test_texts = [
        "BTC mooning! Bulls are back, ATH incoming!",
        "Getting rekt on this dump, total capitulation",
        "Diamond hands holding strong through the dip",
        "This looks like a rug pull, exit scam vibes",
        "Accumulation phase, whales are loading up",
        "Paper hands selling, we're gonna make it WAGMI",
        "Bitcoin consolidating, neutral market conditions",
    ]

    print("Analyzing crypto sentiment...")
    print()

    for text in test_texts:
        result = analyzer.analyze(text)

        # Determine marker based on label
        marker = "[BULL]" if result['label'] == 'bullish' else "[BEAR]" if result['label'] == 'bearish' else "[NEUT]"

        print(f"{marker} {result['label'].upper():8} (score: {result['score']:+.2f})")
        print(f"   Text: {text[:50]}...")
        print(f"   VADER: {result['vader_score']:+.2f}", end="")
        if analyzer.finbert is not None:
            print(f" | FinBERT: {result['finbert_score']:+.2f}")
        else:
            print()
        print()

    print("=" * 60)
    print("[+] Hybrid Sentiment Analysis Working")
    print("=" * 60)

if __name__ == '__main__':
    test_crypto_sentiment()
