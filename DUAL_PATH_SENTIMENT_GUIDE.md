# HIMARI Dual-Path Sentiment Integration Guide

## Overview

This guide explains how to integrate your **fine-tuned Financial-RoBERTa model (85% accuracy)** with a **fast Twitter-RoBERTa model** for social media sentiment analysis.

### Why Dual-Path?

| Event Timeline | What Happens | Signal Source |
|----------------|--------------|---------------|
| T+0 sec | Trump tweets | - |
| T+30 sec | Crypto Twitter reacts | **Twitter-RoBERTa (ALERT)** |
| T+2 min | Liquidation cascade begins | $2B+ liquidations |
| T+10 min | Bloomberg/Reuters reports | **Your Fine-tuned model (TRADE)** |

If you only use news sentiment, you're **10 minutes behind** the move.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         TEXT INPUT              │
                    │         + source                │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │   FAST PATH       │           │   ACCURATE PATH   │
        │   < 5ms           │           │   < 50ms          │
        │                   │           │                   │
        │ Twitter-RoBERTa   │           │ Your Fine-tuned   │
        │ cardiffnlp/...    │           │ Financial-RoBERTa │
        │                   │           │                   │
        │ Sources:          │           │ Sources:          │
        │ - Twitter         │           │ - Bloomberg       │
        │ - Reddit          │           │ - Reuters         │
        │ - Telegram        │           │ - CoinDesk        │
        │ - Discord         │           │ - News            │
        │                   │           │                   │
        │ Signal: ALERT     │           │ Signal: TRADE     │
        │ (early warning)   │           │ (act on it)       │
        └───────────────────┘           └───────────────────┘
```

---

## Installation

### Step 1: Install your fine-tuned model

Extract your `financial-roberta-crypto-finetuned.zip` to the models directory:

```bash
# Create models directory
mkdir -p models/financial-roberta-crypto-finetuned

# Extract fine-tuned model
unzip financial-roberta-crypto-finetuned.zip -d models/financial-roberta-crypto-finetuned/
```

### Step 2: Set environment variable

```bash
export HIMARI_FINE_TUNED_MODEL_PATH="./models/financial-roberta-crypto-finetuned"
```

Or in `.env`:

```
HIMARI_FINE_TUNED_MODEL_PATH=./models/financial-roberta-crypto-finetuned
```

### Step 3: Verify installation

```bash
python test_dual_path_sentiment.py
```

---

## Usage

### Basic Usage

```python
from primitives.dual_path_sentiment import create_dual_path_analyzer

# Create analyzer (auto-detects fine-tuned model)
analyzer = create_dual_path_analyzer()

# Social media → Fast path → ALERT signal
result = analyzer.analyze(
    "BTC mooning! 🚀🚀🚀",
    source="twitter"
)
print(f"Signal: {result.signal_type.value}")  # "alert"
print(f"Latency: {result.latency_ms:.1f}ms")  # ~3-5ms

# News → Accurate path → TRADE signal
result = analyzer.analyze(
    "Bitcoin surges past $100,000 as institutional demand grows",
    source="bloomberg"
)
print(f"Signal: {result.signal_type.value}")  # "trade"
print(f"Latency: {result.latency_ms:.1f}ms")  # ~40ms
```

### Integration with SocialSentimentAggregator

```python
from primitives.dual_path_sentiment import create_dual_path_analyzer
from primitives.social_sentiment_aggregator import SocialSentimentAggregator, SocialPost
from datetime import datetime

analyzer = create_dual_path_analyzer()
aggregator = SocialSentimentAggregator()

# Process incoming social post
text = "BTC looking strong today!"
source = "twitter"

# Analyze with dual-path
result = analyzer.analyze(text, source=source)

# Create post with sentiment
post = SocialPost(
    id="tweet_123",
    source=source,
    symbol="BTCUSDT",
    text=text,
    timestamp=datetime.now(),
    author_id="whale_trader",
    engagement_score=1500.0,  # High engagement
    sentiment_score=result.score
)

# Add to aggregator
aggregator.add_post(post)

# Get rolling aggregates
features = aggregator.get_aggregates("BTCUSDT")
# {'twitter_15m_mean': 0.65, 'twitter_15m_volume': 50, ...}
```

### Redis Signal Publishing

```python
import redis
from primitives.dual_path_sentiment import create_dual_path_analyzer

r = redis.Redis()
analyzer = create_dual_path_analyzer()

def publish_sentiment(text: str, source: str, symbol: str):
    result = analyzer.analyze(text, source=source)
    
    # Store in Redis
    r.hset(f"signals:{symbol}:latest", mapping={
        "sentiment_score": result.score,
        "sentiment_label": result.label,
        "sentiment_signal": result.signal_type.value,
        "sentiment_confidence": result.confidence,
        "sentiment_path": result.path_used,
        "sentiment_latency_ms": result.latency_ms
    })
    
    # Publish alert for fast path (early warning)
    if result.signal_type.value == "alert":
        r.publish(f"alerts:{symbol}", f"SENTIMENT_ALERT:{result.label}")
    
    return result
```

---

## Configuration

### DualPathConfig Options

```python
from primitives.dual_path_sentiment import DualPathConfig, DualPathSentimentAnalyzer

config = DualPathConfig(
    # Model paths
    fast_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    accurate_model="soleimanian/financial-roberta-large-sentiment",
    fine_tuned_model_path="./models/financial-roberta-crypto-finetuned",
    
    # Use fine-tuned model (set True after training)
    use_fine_tuned=True,
    
    # Confidence thresholds
    fast_confidence_threshold=0.60,    # Lower for alerts
    accurate_confidence_threshold=0.70, # Higher for trades
    
    # Latency targets (ms)
    fast_latency_target=5.0,
    accurate_latency_target=50.0,
    
    # Score thresholds
    bullish_threshold=0.3,
    bearish_threshold=-0.3,
    
    # Device
    device="cuda",  # or "cpu"
    
    # Caching
    cache_ttl_seconds=60,
    
    # Fallback behavior
    fallback_to_fast_on_timeout=True,
    accurate_timeout_ms=100.0
)

analyzer = DualPathSentimentAnalyzer(config)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIMARI_FINE_TUNED_MODEL_PATH` | Path to fine-tuned model | `./models/financial-roberta-crypto-finetuned` |
| `HIMARI_SENTIMENT_ENABLED` | Enable sentiment module | `false` |
| `HIMARI_ENHANCED_LAYER1_ENABLED` | Enable enhanced Layer 1 | `false` |

---

## Performance

### Expected Latencies

| Path | Model | Target | Typical | GPU |
|------|-------|--------|---------|-----|
| Fast | Twitter-RoBERTa | <5ms | 3-8ms | 2-3ms |
| Accurate | Fine-tuned Financial-RoBERTa | <50ms | 30-60ms | 15-25ms |

### Accuracy

| Model | Domain | Accuracy | Directional Acc |
|-------|--------|----------|-----------------|
| Twitter-RoBERTa | Social media | 70-75% | ~65% |
| Your Fine-tuned | News headlines | **85.1%** | **85.0%** |

---

## Signal Types

| Signal | Meaning | Action |
|--------|---------|--------|
| `ALERT` | Early warning from social media | Monitor closely, don't trade yet |
| `TRADE` | High-confidence from news | Can act on signal |
| `INFO` | Low confidence | Informational only |

### Trading Strategy Example

```python
def process_sentiment_signal(result):
    if result.signal_type.value == "alert":
        # Fast path alert - early warning
        if result.label == "bearish" and result.confidence > 0.7:
            # High-confidence bearish alert from social
            reduce_position_size(factor=0.5)
            set_tighter_stops()
        
    elif result.signal_type.value == "trade":
        # Accurate path - act on it
        if result.label == "bullish" and result.confidence > 0.8:
            enter_long()
        elif result.label == "bearish" and result.confidence > 0.8:
            enter_short()
```

---

## Files Added

| File | Purpose |
|------|---------|
| `primitives/dual_path_sentiment.py` | Main dual-path analyzer |
| `test_dual_path_sentiment.py` | Integration tests |
| `DUAL_PATH_SENTIMENT_GUIDE.md` | This guide |

---

## Troubleshooting

### Model not loading

```
Failed to load accurate model: ...
```

**Solution:** Ensure the fine-tuned model directory contains:
- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json`
- `vocab.txt` or `tokenizer_config.json`

### Slow latency

**Solution:** 
1. Use GPU: `device="cuda"`
2. Reduce batch size
3. Use model quantization (INT8)

### Out of memory

**Solution:**
1. Use CPU for one model: `device="cpu"`
2. Load models sequentially
3. Clear cache: `analyzer._cache.clear()`

---

## Next Steps

1. ✅ Fine-tune model (DONE - 85% accuracy)
2. ⬜ Deploy to production
3. ⬜ Paper trade for 2-4 weeks
4. ⬜ Monitor directional accuracy live
5. ⬜ Tune confidence thresholds based on results
6. ⬜ Add more social media sources (Telegram, Discord)
