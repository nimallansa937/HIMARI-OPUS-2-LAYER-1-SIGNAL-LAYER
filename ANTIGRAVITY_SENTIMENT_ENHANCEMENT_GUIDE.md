# HIMARI Sentiment Layer - Advanced Enhancement Implementation Guide

**For**: Antigravity Coding Agent
**Date**: 2024-12-24
**Version**: 1.0
**Status**: IMPLEMENTATION SPECIFICATION

---

## EXECUTIVE SUMMARY

This guide provides complete specifications for implementing 7 advanced sentiment analysis enhancements to the existing HIMARI Enhanced Layer 1 Signal System. The current system achieves +6.05 Sharpe improvement with basic VADER + FinBERT hybrid sentiment (static 35/65 weighting). These enhancements target an additional +4-7% Sharpe improvement through:

1. Temporal lag feature engineering
2. Dynamic regime-adaptive weighting
3. Production latency validation gates
4. Fine-tuning strategy with ROI analysis
5. Multi-source social media integration
6. Comprehensive monitoring infrastructure
7. Multi-asset expansion roadmap

**Current Baseline Performance**:
- Sharpe Ratio: 3.27 (baseline: -2.78)
- Sentiment Accuracy: 87-89% on crypto phrases
- Latency: 0.21ms average (well under 10ms target)
- Components: 7 primitives + IntegratedSignalLayer operational

**Target Post-Enhancement Performance**:
- Sharpe Ratio: 3.4-3.5 (+4-7% improvement)
- Sentiment Accuracy: 92-95% (with fine-tuning)
- Latency: p99 < 50ms (with benchmarking gates)
- Multi-source coverage: News + Twitter + Reddit

---

## CURRENT SYSTEM ARCHITECTURE

### Existing Components (DO NOT MODIFY)

**File: primitives/hybrid_sentiment.py (328 lines)**
- `HybridSentimentAnalyzer` class
- VADER lexicon with 24 crypto-specific terms
- FinBERT (ProsusAI/finbert) transformer model
- Static weighting: 35% VADER + 65% FinBERT
- Batch processing support (max 32 texts)
- Current accuracy: 87-89%

**File: primitives/integrated_signal_layer.py (363 lines)**
- `IntegratedSignalLayer` class (master integration)
- SRM risk gating via Redis (thresholds: 0.5, 0.7, 0.9)
- Lines 159-172: Sentiment initialization with lazy import
- Sentiment currently optional (config.sentiment_enabled flag)

**File: config.py (EnhancedSignalConfig dataclass, lines 210-329)**
- `sentiment_enabled: bool = False` (currently disabled)
- `sentiment_vader_weight: float = 0.35` (static)
- `sentiment_finbert_weight: float = 0.65` (static)
- `sentiment_model: str = "ProsusAI/finbert"`
- `sentiment_batch_size: int = 32`

### Current Data Flow

```
OHLCV Data → IntegratedSignalLayer.update()
    ↓
StreamingIndicators (RSI, MACD, BB, ATR, EMA)
    ↓
MultiHorizonMomentum (5, 10, 21, 63 bars)
    ↓
OrderBookImbalance (academic OBI formula)
    ↓
StreamingHMM (regime detection: Bull/Bear/Range)
    ↓
RegimeAwareSignalFusion (regime-adaptive weighting)
    ↓
[OPTIONAL] HybridSentimentAnalyzer
    ↓
SRM Risk Gating (Redis: srm:risk:{symbol})
    ↓
IntegratedSignalOutput (composite_signal, position_multiplier, etc.)
```

---

## ENHANCEMENT 1: SENTIMENT LAG FEATURES

### Priority: ⭐⭐⭐ CRITICAL
### Complexity: LOW
### Impact: +2-5% Sharpe
### Evidence Base: 0.806 Pearson correlation with 3-hour lagged sentiment

### Problem Statement

Current implementation uses ONLY instantaneous sentiment (current bar). Research shows crypto price movements correlate with **lagged sentiment** at optimal horizons:
- **News sentiment**: 0-3 hour lag (strongest at 1.5 hours)
- **Twitter sentiment**: 15min-2 hour lag (strongest at 45 minutes)
- **Reddit sentiment**: 3-6 hour lag (strongest at 4.5 hours)

Missing lag features means we ignore 80% of predictive signal from sentiment data.

### Technical Specification

**New File to Create**: `primitives/sentiment_lag_buffer.py`

**Core Component**: `SentimentLagBuffer` class

**Purpose**: Maintain rolling buffer of sentiment scores across multiple time horizons for each symbol.

**Requirements**:

1. **Buffer Storage**:
   - Store last N sentiment scores per symbol (N = max_lag_bars + 1)
   - Default N = 360 bars (6 hours at 1-minute bars)
   - Use collections.deque for O(1) append/pop operations
   - Memory footprint: ~3KB per symbol for 360 floats

2. **Lag Configuration**:
   ```
   DEFAULT_LAG_HORIZONS = {
       'news': [30, 60, 90, 120, 180],      # minutes: 30min to 3h
       'twitter': [15, 30, 45, 60, 120],    # minutes: 15min to 2h
       'reddit': [180, 240, 270, 300, 360]  # minutes: 3h to 6h
   }
   ```

3. **Feature Engineering Methods**:
   - `get_lag_features(symbol, bar_interval_minutes)` → Dict[str, float]
   - Returns features like:
     - `news_lag_30m`, `news_lag_60m`, `news_lag_90m`, etc.
     - `twitter_lag_15m`, `twitter_lag_30m`, etc.
     - `reddit_lag_180m`, `reddit_lag_240m`, etc.
   - Handle missing data: return 0.0 if buffer insufficient

4. **Efficient Indexing**:
   - Convert time lags to bar indices based on bar_interval
   - Example: For 1-minute bars, 30-minute lag = index -30
   - For 5-minute bars, 30-minute lag = index -6
   - Use: `lag_index = -(lag_minutes // bar_interval_minutes)`

5. **Aggregation Features** (optional but high-value):
   - `sentiment_acceleration`: (current - lag_30m) / 30
   - `sentiment_momentum_3h`: sum of last 180 minute bars
   - `sentiment_reversal_flag`: bool if sentiment flipped sign in last hour

### Integration Points

**Modify File**: `primitives/hybrid_sentiment.py`

**Changes Required**:

1. Add to `HybridSentimentConfig` dataclass:
   ```
   enable_lag_features: bool = True
   max_lag_bars: int = 360
   lag_horizons: Dict[str, List[int]] = field(default_factory=...)
   bar_interval_minutes: int = 1
   ```

2. Add to `HybridSentimentAnalyzer.__init__()`:
   - Initialize `self.lag_buffer = SentimentLagBuffer(config)`

3. Modify `analyze()` method:
   - After computing composite score, call `self.lag_buffer.update(symbol, score)`
   - Return lag features in output dict: `result['lag_features'] = self.lag_buffer.get_lag_features(symbol)`

**Modify File**: `primitives/integrated_signal_layer.py`

**Integration into Fusion** (lines 273-280):

1. When calling `self.fusion.fuse(components, price_return)`:
   - Extract lag features from sentiment output
   - Add each lag feature as separate component signal
   - Register lag signals with base_weight=0.8 (slightly lower than primary sentiment)

2. Example component additions:
   ```
   components['sentiment_current'] = sentiment_result['score']
   components['sentiment_lag_30m'] = lag_features.get('news_lag_30m', 0.0)
   components['sentiment_lag_60m'] = lag_features.get('news_lag_60m', 0.0)
   components['sentiment_acceleration'] = lag_features.get('sentiment_acceleration', 0.0)
   ```

### Validation Criteria

**Unit Tests** (new file: `tests/test_sentiment_lags.py`):

1. Test buffer overflow handling (add 400 scores to 360-bar buffer)
2. Test lag indexing accuracy (verify 30-minute lag = correct historical score)
3. Test multi-symbol isolation (BTCUSDT buffer doesn't affect ETHUSDT)
4. Test aggregation feature calculations
5. Test graceful degradation (return 0.0 when buffer insufficient)

**CPCV Validation** (modify `scripts/run_cpcv_validation.py`):

1. Run baseline vs enhanced+lags comparison
2. Expected improvement: +0.2 to +0.5 Sharpe
3. **CRITICAL**: If improvement < +0.15 Sharpe, investigate correlation analysis
4. Generate lag correlation heatmap (sentiment_lag vs future_returns)

**Success Metrics**:
- Sharpe improvement: +0.2 minimum, +0.5 target
- Correlation at optimal lag > 0.7
- No latency increase (< 0.5ms additional)
- Memory usage < 5MB for 100 symbols

---

## ENHANCEMENT 2: DYNAMIC REGIME-BASED WEIGHTING

### Priority: ⭐⭐ HIGH
### Complexity: MEDIUM
### Impact: +21% accuracy improvement, +2-4% Sharpe
### Evidence Base: Academic paper showing regime-adaptive models outperform static

### Problem Statement

Current system uses **STATIC** weighting: 35% VADER + 65% FinBERT regardless of market conditions. Research shows optimal weighting varies by volatility regime:

- **NORMAL regime** (ATR < 2% daily): VADER 35% / FinBERT 65% (current)
- **HIGH_VOLATILITY regime** (ATR > 4% daily): VADER 20% / FinBERT 50% / Twitter 30%
- **LOW_ENGAGEMENT regime** (social volume < -1σ): VADER 50% / FinBERT 50% / News 0%

Static weighting ignores 21% accuracy gain from regime adaptation.

### Technical Specification

**New File to Create**: `primitives/dynamic_sentiment_weights.py`

**Core Component**: `DynamicSentimentWeighter` class

**Purpose**: Compute optimal sentiment source weights based on current market regime.

**Requirements**:

1. **Regime Detection Inputs**:
   - Volatility regime: Use ATR from `StreamingIndicators`
   - Social engagement regime: Social volume z-score (requires social data feed)
   - Market regime: Reuse `StreamingHMM` regime output (Bull/Bear/Range)

2. **Regime Thresholds** (add to config):
   ```
   VOLATILITY_THRESHOLDS = {
       'low': 0.015,      # ATR < 1.5% daily
       'normal': 0.040,   # ATR 1.5-4% daily
       'high': 0.040      # ATR > 4% daily
   }

   SOCIAL_VOLUME_THRESHOLDS = {
       'low_engagement': -1.0,    # z-score < -1
       'normal': 1.0,             # z-score -1 to +1
       'high_engagement': 1.0     # z-score > +1
   }
   ```

3. **Weight Lookup Table**:
   ```
   REGIME_WEIGHT_MATRIX = {
       ('NORMAL', 'normal', 'Range'): {
           'vader': 0.35, 'finbert': 0.65, 'twitter': 0.0
       },
       ('HIGH_VOLATILITY', 'high_engagement', 'Bull'): {
           'vader': 0.20, 'finbert': 0.50, 'twitter': 0.30
       },
       ('LOW_VOLATILITY', 'low_engagement', 'Range'): {
           'vader': 0.50, 'finbert': 0.50, 'twitter': 0.0
       },
       # ... define all 27 combinations (3 vol × 3 social × 3 market)
   }
   ```

4. **Fallback Strategy**:
   - If regime combination not in matrix, use closest match via Euclidean distance
   - If all sources unavailable, fallback to current static 35/65
   - Log regime transitions for monitoring

5. **Smoothing** (prevent weight thrashing):
   - Apply EMA smoothing to weight transitions (alpha=0.1)
   - Prevent weight changes > 0.15 per update
   - Require 5 consecutive bars in new regime before switching

### Integration Points

**Modify File**: `primitives/hybrid_sentiment.py`

**Changes Required**:

1. Add to `HybridSentimentConfig`:
   ```
   enable_dynamic_weighting: bool = True
   weight_smoothing_alpha: float = 0.1
   min_regime_duration: int = 5
   weight_change_limit: float = 0.15
   ```

2. Add to `HybridSentimentAnalyzer.__init__()`:
   - Initialize `self.weighter = DynamicSentimentWeighter(config)`
   - Initialize `self.current_weights = {'vader': 0.35, 'finbert': 0.65}`

3. Modify `analyze()` method signature:
   ```
   def analyze(self, text, regime_context=None):
       # regime_context = {
       #     'volatility_regime': 'NORMAL',
       #     'social_regime': 'normal',
       #     'market_regime': 'Bull',
       #     'atr': 0.025
       # }
   ```

4. Inside `analyze()`:
   - If regime_context provided, call `weights = self.weighter.get_weights(regime_context)`
   - Apply smoothing: `self.current_weights = smooth(self.current_weights, weights)`
   - Use dynamic weights instead of static 0.35/0.65
   - Return weights in output for monitoring: `result['weights_used'] = self.current_weights`

**Modify File**: `primitives/integrated_signal_layer.py`

**Pass Regime Context** (in `update()` method, around line 223):

1. Before calling sentiment analyzer, construct regime context:
   ```
   regime_context = {
       'volatility_regime': self._classify_volatility(ind_values.get('atr')),
       'social_regime': 'normal',  # TODO: Add social volume tracking
       'market_regime': regime,     # From HMM
       'atr': ind_values.get('atr')
   }
   ```

2. Pass to sentiment: `sentiment_result = self.sentiment.analyze(text, regime_context)`

3. Extract and log weights: `logger.debug(f"Sentiment weights: {sentiment_result['weights_used']}")`

**New Method to Add**: `_classify_volatility(atr)` helper

### Validation Criteria

**Unit Tests** (new file: `tests/test_dynamic_weights.py`):

1. Test regime classification (ATR=0.05 → HIGH_VOLATILITY)
2. Test weight lookup (verify correct weights for all 27 regime combinations)
3. Test smoothing (verify EMA applied correctly, no sudden jumps > 0.15)
4. Test fallback (unknown regime → nearest match)
5. Test regime duration filter (5-bar minimum before switch)

**A/B Testing** (add to CPCV validation):

1. Run 3 variants:
   - Baseline: Static 35/65
   - Enhanced: Dynamic weights without smoothing
   - Production: Dynamic weights WITH smoothing
2. Expected: Production variant +21% accuracy over baseline
3. **CRITICAL**: Monitor for weight thrashing (> 10 transitions per hour = BAD)

**Success Metrics**:
- Accuracy improvement: +15% minimum, +21% target
- Sharpe improvement: +0.15 minimum, +0.30 target
- Weight stability: < 5 transitions per hour
- Regime classification accuracy: > 90%

---

## ENHANCEMENT 3: LATENCY VALIDATION BENCHMARKS

### Priority: ⭐ CRITICAL (for production deployment)
### Complexity: LOW
### Impact: Risk mitigation (prevent production failures)
### Evidence Base: p99 latency spikes can trigger SRM emergency halts

### Problem Statement

Current system tracks average latency (0.21ms) but does NOT enforce p99 latency SLAs. Production deployment requires:
- **p99 < 50ms** for signal generation (IntegratedSignalLayer.update())
- **p99 < 100ms** for sentiment analysis (with FinBERT inference)
- **p99 < 200ms** for end-to-end (data ingestion → signal output)

Without validation gates, latency regressions can:
1. Trigger SRM emergency halts (if signal delay > 500ms)
2. Cause stale price data usage (arbitrage risk)
3. Miss optimal entry/exit windows (slippage cost)

### Technical Specification

**New File to Create**: `validation/latency_validator.py`

**Core Component**: `LatencyBenchmark` class

**Purpose**: Measure, validate, and enforce p50/p95/p99 latency SLAs across all components.

**Requirements**:

1. **Metrics Collection**:
   - Use `time.perf_counter()` for nanosecond precision
   - Track latency distributions per component:
     - `sentiment_analysis_latency_ms`
     - `hmm_update_latency_ms`
     - `fusion_latency_ms`
     - `total_signal_latency_ms`
   - Store last 10,000 measurements per metric (rolling window)

2. **Percentile Calculation**:
   - Use numpy.percentile() for p50/p95/p99 computation
   - Recompute percentiles every 100 updates (not every update for efficiency)
   - Return current percentiles via `get_latency_stats()` method

3. **SLA Enforcement** (add to config):
   ```
   LATENCY_SLAS = {
       'sentiment_analysis': {
           'p50': 30,   # milliseconds
           'p95': 80,
           'p99': 100,
           'action_on_breach': 'LOG_WARNING'
       },
       'total_signal': {
           'p50': 5,
           'p95': 20,
           'p99': 50,
           'action_on_breach': 'DISABLE_COMPONENT'  # Fallback to non-sentiment
       }
   }
   ```

4. **Breach Actions**:
   - `LOG_WARNING`: Log to monitoring system, continue operation
   - `DISABLE_COMPONENT`: Temporarily disable slow component (e.g., FinBERT)
   - `RAISE_ALERT`: Send alert to ops team via webhook
   - `EMERGENCY_HALT`: Trigger SRM halt if total latency > 500ms

5. **Quantization Warning System**:
   - **CRITICAL**: INT8 quantization can **increase** CPU latency despite smaller model size
   - Benchmark BEFORE and AFTER quantization
   - If post-quantization p99 > pre-quantization p95, REJECT quantization
   - Log warning: "Quantization increased latency - reverting to FP32"

### Integration Points

**Modify File**: `primitives/integrated_signal_layer.py`

**Add Latency Instrumentation** (in `update()` method):

1. Import latency validator at module level:
   ```
   from validation.latency_validator import LatencyBenchmark
   ```

2. Initialize in `__init__()`:
   ```
   self.latency_bench = LatencyBenchmark(config.latency_slas)
   ```

3. Wrap component calls with timing:
   ```
   # Example for sentiment
   with self.latency_bench.measure('sentiment_analysis'):
       sentiment_result = self.sentiment.analyze(text, regime_context)

   # Example for total signal
   with self.latency_bench.measure('total_signal'):
       # ... entire update() method body ...
   ```

4. Check SLA compliance every N updates:
   ```
   if self.update_count % 100 == 0:
       stats = self.latency_bench.get_latency_stats()
       breaches = self.latency_bench.check_sla_breaches()
       for breach in breaches:
           self._handle_latency_breach(breach)
   ```

**Modify File**: `config.py`

**Add Latency Configuration**:

1. Add to `EnhancedSignalConfig`:
   ```
   latency_slas: Dict[str, Dict[str, Any]] = field(default_factory=...)
   latency_check_interval: int = 100  # Check every N updates
   enable_latency_gating: bool = True
   quantization_latency_tolerance: float = 1.2  # Max 20% slowdown
   ```

### Validation Criteria

**Benchmark Script** (new file: `scripts/benchmark_latency.py`):

1. Generate 10,000 synthetic OHLCV updates
2. Measure p50/p95/p99 for each component
3. Generate latency distribution histograms
4. Test quantized vs non-quantized FinBERT
5. Output validation report:
   ```
   Component            p50    p95    p99    SLA    Status
   ----------------------------------------------------------
   HMM Update          0.05   0.12   0.18   1.0    [PASS]
   Sentiment (FP32)    25.3   72.1   98.5   100    [PASS]
   Sentiment (INT8)    31.2   89.3   142.1  100    [FAIL]  ← REJECT
   Total Signal        1.8    4.2    8.7    50     [PASS]
   ```

**Load Testing** (add to CPCV validation):

1. Simulate 1000 updates/second sustained load
2. Monitor latency degradation under load
3. **CRITICAL**: p99 should not exceed 2x p99 at idle load
4. If p99 > 100ms under load, REDUCE batch size or DISABLE sentiment

**Success Metrics**:
- p99 total signal latency < 50ms (idle)
- p99 total signal latency < 100ms (under 1000 updates/sec load)
- Zero SLA breaches in 72-hour shadow mode test
- Quantization acceptance rate: Only if latency improvement OR < 20% slowdown

---

## ENHANCEMENT 4: FINE-TUNING STRATEGY WITH ROI ANALYSIS

### Priority: ⭐ HIGH
### Complexity: MEDIUM-HIGH
### Impact: +15% accuracy, ROI 2800-6000% annually
### Evidence Base: 500 labeled crypto headlines → +15% accuracy gain

### Problem Statement

Current FinBERT model is pre-trained on **general financial news** (earnings reports, stock market). Crypto domain has unique vocabulary and sentiment patterns:
- "HODL", "WAGMI", "rekt", "mooning" = not in FinBERT vocabulary
- Bitcoin-specific catalysts (halving, ETF approvals) ≠ stock catalysts
- Current accuracy: 87-89% → with fine-tuning: 92-95%

**Economic Case**:
- Labeling cost: 500 examples × $2.40/example = **$1,200**
- Time investment: ~40 hours (data collection + training + validation)
- Expected Sharpe improvement: +0.2 to +0.4
- Annual alpha improvement: $28,000 to $72,000 (on $100K portfolio)
- **ROI: 2800% to 6000% annually**

### Technical Specification

**Phase 1: Data Collection & Labeling**

**New Directory**: `data/fine_tuning/`

**Required Files**:
1. `crypto_headlines_raw.jsonl` - Raw scraped headlines
2. `crypto_headlines_labeled.jsonl` - Human-labeled dataset
3. `label_distribution.json` - Class balance report

**Data Collection Strategy**:

1. **Sources** (target 1000 headlines, label 500):
   - CoinDesk API: 300 headlines (2022-2024)
   - CryptoCompare news feed: 300 headlines
   - Twitter @binance, @coinbase: 200 tweets
   - Reddit r/cryptocurrency top posts: 200 posts

2. **Sampling Strategy** (to ensure class balance):
   - 40% bullish examples (keywords: "rally", "surge", "ATH", "bull run")
   - 40% bearish examples (keywords: "crash", "dump", "selloff", "capitulation")
   - 20% neutral examples (keywords: "consolidation", "sideways", "mixed")

3. **Labeling Guidelines** (provide to labelers):
   ```
   Label: BULLISH (1)
   - Price-positive language ("Bitcoin surges", "Ethereum rally")
   - Adoption news ("PayPal integrates crypto")
   - Regulatory approval ("SEC approves ETF")

   Label: BEARISH (-1)
   - Price-negative language ("Bitcoin crashes", "selloff accelerates")
   - Security incidents ("Exchange hacked")
   - Regulatory crackdown ("China bans mining")

   Label: NEUTRAL (0)
   - Factual reporting without sentiment ("Bitcoin trades at $45K")
   - Mixed signals ("Bitcoin up but altcoins down")
   - Neutral announcements ("Coinbase lists new token")
   ```

4. **Quality Control**:
   - Use 3 labelers per headline
   - Require 2/3 agreement for inclusion
   - Measure inter-annotator agreement (Cohen's Kappa > 0.7 required)
   - Discard headlines with < 66% agreement

**Phase 2: Fine-Tuning Pipeline**

**New File**: `scripts/fine_tune_finbert.py`

**Requirements**:

1. **Training Configuration**:
   ```
   FINE_TUNING_CONFIG = {
       'base_model': 'ProsusAI/finbert',
       'learning_rate': 2e-5,
       'batch_size': 16,
       'num_epochs': 3,
       'warmup_steps': 100,
       'weight_decay': 0.01,
       'max_seq_length': 128,
       'train_split': 0.8,
       'val_split': 0.1,
       'test_split': 0.1
   }
   ```

2. **Data Augmentation** (optional, to expand 500 → 1500 examples):
   - Synonym replacement (80% original, 20% augmented)
   - Back-translation (English → Spanish → English)
   - Paraphrasing with GPT-4 (careful to preserve sentiment)

3. **Training Process**:
   - Use Hugging Face Trainer API
   - Monitor validation loss every 50 steps
   - Save checkpoint every epoch
   - Early stopping if val_loss increases for 2 consecutive checks

4. **Evaluation Metrics**:
   - Accuracy on test set (target > 92%)
   - F1-score per class (target > 0.90)
   - Confusion matrix (identify if bias toward bull/bear)
   - Compare vs baseline FinBERT on same test set

**Phase 3: Model Deployment**

**File to Modify**: `primitives/hybrid_sentiment.py`

**Changes Required**:

1. Add to `HybridSentimentConfig`:
   ```
   finbert_model: str = "ProsusAI/finbert"  # Default
   use_fine_tuned: bool = False
   fine_tuned_model_path: str = "./models/finbert-crypto-finetuned"
   ```

2. Modify `__init__()` to support fine-tuned model:
   ```
   if config.use_fine_tuned and os.path.exists(config.fine_tuned_model_path):
       logger.info(f"Loading FINE-TUNED FinBERT from {config.fine_tuned_model_path}")
       self.finbert = pipeline("sentiment-analysis", model=config.fine_tuned_model_path)
   else:
       logger.info("Loading baseline FinBERT")
       self.finbert = pipeline("sentiment-analysis", model=config.finbert_model)
   ```

3. Add model version tracking:
   ```
   def get_model_info(self):
       return {
           'model_type': 'fine_tuned' if self.config.use_fine_tuned else 'baseline',
           'model_path': self.finbert.model.name_or_path,
           'version': '1.0.0'  # Track fine-tuning version
       }
   ```

**Phase 4: Active Learning Pipeline** (Optional - Future Enhancement)

**Purpose**: Continuously improve model with production data

**Strategy**:
1. Identify low-confidence predictions (0.4 < prob < 0.6)
2. Flag for human review (5 examples/day = $12/day = $4,400/year)
3. Retrain quarterly with accumulated labels (500 + 1,825 new = 2,325 total)
4. Expected accuracy gain: +1-2% per quarter

**Implementation**:
- Add `confidence_threshold_for_review` = 0.6 to config
- Log low-confidence examples to `data/active_learning_queue.jsonl`
- Schedule quarterly retraining job

### Validation Criteria

**Pre-Training Validation**:

1. **Dataset Quality Check** (new file: `scripts/validate_labels.py`):
   - Class distribution: 35-45% bullish, 35-45% bearish, 15-25% neutral
   - No duplicate headlines (use fuzzy matching)
   - Average headline length: 10-30 tokens
   - No HTML artifacts or special characters
   - Cohen's Kappa > 0.7 for inter-annotator agreement

2. **Baseline Benchmark**:
   - Run current FinBERT on labeled test set
   - Record baseline accuracy (expected ~87%)
   - This is the "before" metric

**Post-Training Validation**:

1. **Model Performance** (run `scripts/evaluate_finetuned.py`):
   - Test set accuracy > 92% (target)
   - Per-class F1-score > 0.90
   - Confusion matrix: < 10% misclassification rate
   - **CRITICAL**: No catastrophic forgetting (accuracy on general financial news should remain > 80%)

2. **CPCV Re-validation**:
   - Re-run full CPCV validation with fine-tuned model
   - Expected Sharpe improvement: +0.2 to +0.4 over baseline
   - If improvement < +0.1, investigate label quality or model overfitting

3. **A/B Testing in Shadow Mode**:
   - Run baseline vs fine-tuned models in parallel for 7 days
   - Compare signal divergence rate (expected < 15% divergence)
   - Monitor accuracy drift on live data

**ROI Tracking**:

1. **Cost Tracking**:
   - Labeling cost: $1,200 (one-time)
   - Training compute: ~$50 (3 epochs on A100)
   - Validation time: 40 hours engineer time
   - Total investment: ~$3,000

2. **Benefit Measurement**:
   - Track Sharpe improvement attribution (expected +0.2 to +0.4)
   - Calculate alpha in dollar terms (Sharpe × volatility × √252 × portfolio)
   - Quarterly review: Is ROI maintaining > 1000%?

**Success Metrics**:
- Test set accuracy: > 92%
- Sharpe improvement: +0.2 minimum, +0.4 target
- ROI: > 2000% in first year
- No catastrophic forgetting (general finance accuracy > 80%)
- Deployment readiness: < 2 weeks from label completion to production

---

## ENHANCEMENT 5: SOCIAL MEDIA SENTIMENT INTEGRATION

### Priority: MEDIUM
### Complexity: HIGH
### Impact: +1-3% Sharpe (via crowd behavior signals)
### Evidence Base: Twitter sentiment leads price by 15-120 minutes in crypto

### Problem Statement

Current system uses ONLY news sentiment (FinBERT). Missing social media = missing early warning signals:
- **Twitter**: Retail crowd behavior (FOMO/panic indicators)
- **Reddit**: Discussion depth and quality (conviction indicators)
- **Telegram**: Pump-and-dump coordination signals (fraud detection)

Social media sentiment is LEADING indicator (15-120 min ahead) vs news (0-180 min lag).

### Technical Specification

**Phase 1: Data Pipeline Setup**

**New File**: `data_ingestion/social_media_connector.py`

**Core Component**: `SocialMediaConnector` class

**Purpose**: Stream real-time social media posts for sentiment analysis

**Data Sources**:

1. **Twitter API v2** (Essential Tier - $100/month):
   - Filtered stream: Track keywords ("BTC", "Bitcoin", "crypto", etc.)
   - Volume: ~1,000 tweets/hour for major cryptos
   - Fields: text, created_at, author_id, public_metrics (likes, retweets)
   - Rate limits: 50 requests/15min

2. **Reddit API** (Free tier sufficient):
   - Subreddits: r/cryptocurrency, r/bitcoin, r/ethereum
   - Endpoints: /r/{sub}/new.json (new posts), /r/{sub}/comments (discussions)
   - Volume: ~200 posts/hour
   - Rate limits: 60 requests/minute

3. **Telegram** (Optional - requires custom scraper):
   - Target channels: @bitcoin, @cryptocurrencynews
   - Higher spam risk - use with caution
   - Consider ONLY if Twitter/Reddit insufficient

**Data Schema** (standardize across sources):
```
{
    'id': str,
    'source': 'twitter' | 'reddit' | 'telegram',
    'symbol': str,  # Extracted via regex
    'text': str,
    'timestamp': datetime,
    'author_id': str,
    'engagement_score': float,  # (likes + retweets + comments) normalized
    'verified_author': bool
}
```

**Spam Filtering** (CRITICAL - social media has 70%+ noise):

1. **Author Reputation Filter**:
   - Twitter: Require > 100 followers OR verified account
   - Reddit: Require > 500 karma
   - Remove posts from accounts < 30 days old

2. **Content Quality Filter**:
   - Remove posts < 10 words (likely spam)
   - Remove posts with > 3 emojis per 10 words
   - Remove posts with URLs to suspicious domains (pump-and-dump sites)
   - Remove duplicates (exact text match within 1 hour window)

3. **Engagement Threshold**:
   - Twitter: Require > 5 likes OR > 2 retweets
   - Reddit: Require > 3 upvotes
   - Rationale: Engagement = signal quality proxy

**Expected Retention Rate**: 15-30% of raw posts after filtering

**Phase 2: Twitter-Specific Sentiment Model**

**New Dependency**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Why Separate Model**:
- FinBERT trained on formal news (WSJ, Bloomberg)
- Twitter has informal language, slang, abbreviations
- Twitter-RoBERTa trained on 124M tweets → better accuracy

**Implementation**:

**Modify File**: `primitives/hybrid_sentiment.py`

**Add Third Model to Ensemble**:

1. Add to `HybridSentimentConfig`:
   ```
   enable_social_sentiment: bool = False
   twitter_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
   social_weight: float = 0.0  # Dynamic - set by regime weighter
   ```

2. Add to `__init__()`:
   ```
   if config.enable_social_sentiment:
       self.twitter_roberta = pipeline(
           "sentiment-analysis",
           model=config.twitter_model,
           device=-1  # CPU inference
       )
   ```

3. Modify `analyze()` to support source parameter:
   ```
   def analyze(self, text, regime_context=None, source='news'):
       # source = 'news' | 'twitter' | 'reddit'

       vader_score = self.vader.polarity_scores(text)['compound']

       if source == 'twitter' and self.twitter_roberta:
           transformer_score = self._run_twitter_roberta(text)
       else:
           transformer_score = self._run_finbert(text)

       # Apply regime-based weighting (from Enhancement 2)
       weights = self._get_dynamic_weights(regime_context, source)

       composite = (
           weights['vader'] * vader_score +
           weights['transformer'] * transformer_score
       )
   ```

**Phase 3: Aggregation Strategy**

**Problem**: Can't run sentiment on 1000 tweets/hour individually (too slow)

**Solution**: Batch aggregation with temporal windowing

**New File**: `primitives/social_sentiment_aggregator.py`

**Core Component**: `SocialSentimentAggregator` class

**Requirements**:

1. **Temporal Windows**:
   - 15-minute rolling window (for Twitter - fast-moving)
   - 1-hour rolling window (for Reddit - slower discussions)
   - Compute aggregate sentiment per symbol per window

2. **Aggregation Methods**:
   - **Volume-weighted mean**: `sum(sentiment × engagement) / sum(engagement)`
   - **Median** (robust to outliers)
   - **Extreme sentiment ratio**: `(bullish_count - bearish_count) / total_count`

3. **Output Features** (per symbol):
   ```
   {
       'twitter_15m_mean': float,
       'twitter_15m_volume': int,  # Number of tweets
       'twitter_15m_extreme_ratio': float,
       'reddit_1h_mean': float,
       'reddit_1h_volume': int,
       'reddit_1h_median': float
   }
   ```

4. **Volume Normalization**:
   - Use WelfordOnlineStats to z-score normalize tweet volumes
   - High volume + bullish sentiment = strong signal
   - High volume + neutral sentiment = noise (ignore)

**Phase 4: Integration into Signal Layer**

**Modify File**: `primitives/integrated_signal_layer.py`

**Add Social Sentiment Component** (in `update()` method):

1. Initialize aggregator in `__init__()`:
   ```
   self.social_agg = SocialSentimentAggregator(config) if config.enable_social_sentiment else None
   ```

2. Fetch social features in `update()`:
   ```
   if self.social_agg:
       social_features = self.social_agg.get_aggregates(symbol)

       # Add to components for fusion
       if social_features.get('twitter_15m_mean') is not None:
           components['social_twitter_15m'] = social_features['twitter_15m_mean']

       if social_features.get('reddit_1h_mean') is not None:
           components['social_reddit_1h'] = social_features['reddit_1h_mean']
   ```

3. Register social signals with fusion (in `_register_signals()`):
   ```
   self.fusion.register_signal('social_twitter_15m', 'sentiment', base_weight=1.0)
   self.fusion.register_signal('social_reddit_1h', 'sentiment', base_weight=0.8)
   ```

### Validation Criteria

**Data Quality Validation**:

1. **Spam Filter Effectiveness** (new file: `scripts/validate_social_data.py`):
   - Manual review of 200 random filtered posts
   - Spam rate should be < 5%
   - Relevance rate (actually about crypto) > 90%

2. **Volume Consistency**:
   - Twitter: 500-1500 tweets/hour (after filtering)
   - Reddit: 100-300 posts/hour (after filtering)
   - If volume < 50% of expected, investigate API issues

**Model Performance**:

1. **Twitter-RoBERTa Accuracy**:
   - Benchmark on 200 hand-labeled crypto tweets
   - Target accuracy > 85% (lower than FinBERT OK - informal text harder)
   - Compare vs FinBERT on same tweets (expect +10-15% accuracy)

2. **Lead-Lag Analysis** (CRITICAL):
   - Compute cross-correlation between twitter_sentiment and future_returns
   - Expected optimal lag: 15-120 minutes
   - If no significant correlation at any lag < 6 hours, social signals NOT useful

**CPCV Validation**:

1. **Incremental Testing**:
   - Baseline: No social sentiment
   - Enhanced: With Twitter 15m
   - Full: With Twitter 15m + Reddit 1h
   - Expected Sharpe improvement: +0.1 to +0.3

2. **Feature Importance Analysis**:
   - Use SHAP values to measure social sentiment contribution
   - If feature importance < 5%, consider disabling (not worth latency cost)

**Success Metrics**:
- Sharpe improvement: +0.1 minimum, +0.3 target
- Lead-lag correlation > 0.4 at optimal lag
- Spam rate < 5% post-filtering
- Twitter-RoBERTa accuracy > 85%
- Data pipeline uptime > 99.5%

---

## ENHANCEMENT 6: PRODUCTION MONITORING INFRASTRUCTURE

### Priority: ⭐ CRITICAL (for production deployment)
### Complexity: MEDIUM
### Impact: Risk mitigation + operational visibility
### Evidence Base: Production systems without monitoring fail silently

### Problem Statement

Current system has NO production monitoring:
- Cannot detect accuracy degradation in real-time
- Cannot measure latency under production load
- Cannot track model drift or data quality issues
- Cannot alert on system failures before user impact

Production deployment without monitoring = flying blind at 500mph.

### Technical Specification

**Phase 1: Metrics Instrumentation**

**New File**: `monitoring/metrics_collector.py`

**Core Component**: `PrometheusMetricsCollector` class

**Purpose**: Expose system metrics in Prometheus format for scraping

**Metrics to Track**:

1. **Latency Metrics** (histograms with p50/p95/p99):
   ```
   sentiment_analysis_latency_ms{source="news"}
   sentiment_analysis_latency_ms{source="twitter"}
   hmm_update_latency_ms
   fusion_latency_ms
   total_signal_latency_ms
   ```

2. **Throughput Metrics** (counters):
   ```
   signals_generated_total{symbol="BTCUSDT"}
   sentiment_analyses_total{source="news"}
   sentiment_analyses_total{source="twitter"}
   cache_hits_total
   cache_misses_total
   ```

3. **Accuracy Metrics** (gauges - updated hourly):
   ```
   sentiment_accuracy_percent{source="news", window="1h"}
   sentiment_accuracy_percent{source="twitter", window="1h"}
   signal_sharpe_ratio{symbol="BTCUSDT", window="24h"}
   win_rate_percent{symbol="BTCUSDT", window="24h"}
   ```

4. **Error Metrics** (counters):
   ```
   sentiment_errors_total{error_type="model_timeout"}
   sentiment_errors_total{error_type="invalid_input"}
   redis_connection_errors_total
   srm_gate_activations_total{action="HALT"}
   ```

5. **Data Quality Metrics** (gauges):
   ```
   social_posts_filtered_total{source="twitter", reason="spam"}
   sentiment_confidence_mean{source="news"}
   data_freshness_seconds{source="twitter"}  # Time since last update
   ```

6. **Model Drift Metrics** (gauges - updated daily):
   ```
   sentiment_distribution_skew{source="news"}  # Detect if always bullish/bearish
   regime_transition_rate{period="1h"}
   feature_importance_drift{feature="sentiment_lag_30m"}
   ```

**Implementation**:

1. **Use Prometheus Client Library**:
   ```
   from prometheus_client import Counter, Histogram, Gauge, start_http_server

   # Define metrics
   SIGNAL_LATENCY = Histogram(
       'total_signal_latency_ms',
       'End-to-end signal generation latency',
       buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
   )

   SENTIMENT_ACCURACY = Gauge(
       'sentiment_accuracy_percent',
       'Sentiment prediction accuracy',
       ['source', 'window']
   )
   ```

2. **Instrument Code**:
   - Modify `IntegratedSignalLayer.update()` to record metrics
   - Add `@SIGNAL_LATENCY.time()` decorator to critical methods
   - Update accuracy gauges every 100 updates

3. **Expose Metrics Endpoint**:
   ```
   # Start Prometheus HTTP server on port 8000
   start_http_server(8000)

   # Metrics available at: http://localhost:8000/metrics
   ```

**Phase 2: Grafana Dashboard Setup**

**Dashboard Configuration File**: `monitoring/grafana_dashboard.json`

**6 Required Panels**:

1. **Latency Panel** (Time series graph):
   - Queries:
     ```
     histogram_quantile(0.50, sentiment_analysis_latency_ms)  # p50
     histogram_quantile(0.95, sentiment_analysis_latency_ms)  # p95
     histogram_quantile(0.99, sentiment_analysis_latency_ms)  # p99
     ```
   - Alert: p99 > 100ms for 5 minutes → Page oncall
   - Visualization: Line graph, 1-hour window

2. **Throughput Panel** (Single stat):
   - Query: `rate(signals_generated_total[1m])`
   - Display: Current signals/second
   - Alert: < 10 signals/min for 10 minutes → Warning

3. **Cache Hit Rate Panel** (Gauge):
   - Query: `cache_hits_total / (cache_hits_total + cache_misses_total)`
   - Target: > 80% hit rate
   - Alert: < 60% for 30 minutes → Investigate cache sizing

4. **Accuracy Drift Panel** (Time series):
   - Queries:
     ```
     sentiment_accuracy_percent{source="news", window="1h"}
     sentiment_accuracy_percent{source="twitter", window="1h"}
     ```
   - Baseline overlay: 87% (current accuracy)
   - Alert: < 80% for 2 hours → Model drift detected

5. **Error Rate Panel** (Stacked area chart):
   - Query: `rate(sentiment_errors_total[5m]) by (error_type)`
   - Alert: > 10 errors/min → Critical alert

6. **Data Freshness Panel** (Heatmap):
   - Query: `data_freshness_seconds{source="twitter"}`
   - Target: < 60 seconds
   - Alert: > 300 seconds → Data pipeline stalled

**Phase 3: Alerting Rules**

**Alert Configuration File**: `monitoring/prometheus_alerts.yml`

**Critical Alerts** (page oncall immediately):

1. **Latency SLA Breach**:
   ```
   alert: HighP99Latency
   expr: histogram_quantile(0.99, total_signal_latency_ms) > 100
   for: 5m
   severity: critical
   message: "Signal generation p99 latency exceeded 100ms"
   ```

2. **Accuracy Degradation**:
   ```
   alert: SentimentAccuracyDrop
   expr: sentiment_accuracy_percent{window="1h"} < 75
   for: 2h
   severity: critical
   message: "Sentiment accuracy dropped below 75% - possible model drift"
   ```

3. **Data Pipeline Failure**:
   ```
   alert: SocialDataStale
   expr: data_freshness_seconds{source="twitter"} > 600
   for: 10m
   severity: critical
   message: "Twitter data pipeline stalled - no updates in 10 minutes"
   ```

**Warning Alerts** (notify Slack, no page):

1. **Cache Performance**:
   ```
   alert: LowCacheHitRate
   expr: cache_hits / (cache_hits + cache_misses) < 0.6
   for: 30m
   severity: warning
   ```

2. **Model Confidence Drop**:
   ```
   alert: LowSentimentConfidence
   expr: sentiment_confidence_mean < 0.5
   for: 1h
   severity: warning
   ```

**Phase 4: Accuracy Tracking System**

**Challenge**: How to measure accuracy in production when we don't have labels?

**Solution**: Proxy metrics + periodic manual validation

**New File**: `monitoring/accuracy_tracker.py`

**Proxy Metrics**:

1. **Signal-Return Correlation**:
   - Compute rolling correlation between composite_signal and next_bar_return
   - Expected baseline: > 0.3
   - If drops below 0.2 → accuracy likely degraded

2. **Regime Stability**:
   - Measure regime transition rate (should be stable at ~2-5 per day)
   - If transitions > 20/day → HMM might be unstable

3. **Sentiment Distribution**:
   - Track distribution of sentiment scores (should be roughly normal)
   - If skewed heavily bullish/bearish → model bias or data quality issue

**Manual Validation** (weekly):
- Sample 50 random sentiment analyses
- Human review for correctness
- Track accuracy trend over time
- If accuracy < 80%, trigger model retraining

### Integration Points

**Modify File**: `primitives/integrated_signal_layer.py`

**Add Metrics Collection**:

1. Import metrics collector:
   ```
   from monitoring.metrics_collector import PrometheusMetricsCollector
   ```

2. Initialize in `__init__()`:
   ```
   self.metrics = PrometheusMetricsCollector(config)
   ```

3. Record metrics in `update()`:
   ```
   # Record latency
   self.metrics.record_latency('total_signal', latency_ms)

   # Record throughput
   self.metrics.increment_counter('signals_generated', labels={'symbol': symbol})

   # Update accuracy proxy (signal-return correlation)
   if self.update_count % 100 == 0:
       correlation = self._compute_signal_return_correlation()
       self.metrics.set_gauge('signal_accuracy_proxy', correlation)
   ```

**New Startup Script**: `scripts/start_monitoring.py`

**Purpose**: Launch Prometheus + Grafana containers

```
# Docker Compose approach
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana_dashboard.json:/etc/grafana/provisioning/dashboards/
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Validation Criteria

**Monitoring System Health**:

1. **Metric Ingestion Test**:
   - Generate 1000 synthetic signals
   - Verify all metrics appear in Prometheus
   - Check for missing labels or NaN values

2. **Dashboard Load Test**:
   - Open Grafana dashboard
   - Verify all 6 panels render within 5 seconds
   - Check for query errors or timeouts

3. **Alert Simulation**:
   - Inject latency spike (sleep 200ms)
   - Verify alert fires within 5 minutes
   - Check alert routing (Slack/PagerDuty)

**Operational Readiness**:

1. **Runbook Creation**:
   - Document response procedures for each alert
   - Example: "HighP99Latency → Check CPU usage → Scale up instances OR disable FinBERT"

2. **Shadow Mode Validation**:
   - Run monitoring for 72 hours in shadow mode
   - Verify zero false positive alerts
   - Verify zero missed incidents

**Success Metrics**:
- All 6 Grafana panels operational
- Alert firing latency < 5 minutes
- Zero false positives in 72-hour test
- Metric collection overhead < 1ms per update
- Dashboard query response < 3 seconds

---

## ENHANCEMENT 7: MULTI-ASSET EXPANSION ROADMAP

### Priority: FUTURE (post-BTC validation)
### Complexity: MEDIUM
### Impact: Scalability to 50+ assets
### Evidence Base: Transfer learning works for correlated assets (BTC→ETH: 0.94)

### Problem Statement

Current system is BTC-only. Production needs to support:
- **Tier 1**: BTC, ETH (high correlation = easy)
- **Tier 2**: SOL, BNB, ADA (medium correlation = moderate effort)
- **Tier 3**: Altcoins (low correlation = may need separate models)
- **Future**: Equities, commodities (low correlation = definitely need separate models)

Cannot retrain FinBERT for each asset (too expensive). Need transfer learning strategy.

### Technical Specification

**Phase 1: Asset Correlation Analysis**

**New File**: `analysis/asset_correlation_matrix.py`

**Purpose**: Measure sentiment correlation between assets to determine if transfer learning viable

**Requirements**:

1. **Correlation Metrics to Compute**:
   - Price correlation (Pearson on log returns)
   - Sentiment correlation (on 500 labeled examples per asset)
   - Regime correlation (HMM state alignment)

2. **Decision Matrix**:
   ```
   Correlation > 0.9:  Use BTC model directly (no retraining)
   Correlation 0.7-0.9: Use BTC model + light fine-tuning (100 examples)
   Correlation < 0.7:  Train separate model (500 examples)
   ```

3. **Expected Correlations** (from research):
   - BTC ↔ ETH: 0.94 (use BTC model)
   - BTC ↔ SOL: 0.82 (light fine-tuning)
   - BTC ↔ equities: 0.30-0.45 (separate model)

**Phase 2: Multi-Symbol Configuration**

**Modify File**: `config.py`

**Add Asset-Specific Configs**:

```
@dataclass
class AssetConfig:
    symbol: str
    model_variant: str  # 'btc_base' | 'light_finetuned' | 'dedicated'
    sentiment_sources: List[str]  # ['news', 'twitter', 'reddit']
    lag_horizons: Dict[str, List[int]]
    regime_weights: Dict[str, Dict[str, float]]

ASSET_CONFIGS = {
    'BTCUSDT': AssetConfig(
        symbol='BTCUSDT',
        model_variant='btc_base',
        sentiment_sources=['news', 'twitter', 'reddit'],
        lag_horizons=DEFAULT_LAG_HORIZONS,
        regime_weights=DEFAULT_REGIME_WEIGHTS
    ),
    'ETHUSDT': AssetConfig(
        symbol='ETHUSDT',
        model_variant='btc_base',  # High correlation - reuse BTC model
        sentiment_sources=['news', 'twitter'],
        lag_horizons=DEFAULT_LAG_HORIZONS,
        regime_weights=DEFAULT_REGIME_WEIGHTS  # Same as BTC
    ),
    'SOLUSDT': AssetConfig(
        symbol='SOLUSDT',
        model_variant='light_finetuned',
        sentiment_sources=['news', 'twitter'],
        lag_horizons={'news': [60, 120], 'twitter': [30, 60]},  # Different lags
        regime_weights=SOLANA_REGIME_WEIGHTS  # Custom weights
    )
}
```

**Phase 3: Model Sharing Strategy**

**New File**: `primitives/multi_asset_sentiment_manager.py`

**Core Component**: `MultiAssetSentimentManager` class

**Purpose**: Manage model instances across multiple assets efficiently

**Requirements**:

1. **Model Pooling** (to save memory):
   - Load only 1 instance of BTC base model
   - Share across BTC, ETH, and other high-correlation assets
   - Load separate instances ONLY for dedicated models

2. **Asset Routing Logic**:
   ```
   def get_model_for_asset(self, symbol):
       config = ASSET_CONFIGS[symbol]

       if config.model_variant == 'btc_base':
           return self.btc_base_model  # Shared instance
       elif config.model_variant == 'light_finetuned':
           return self.light_models[symbol]  # Per-asset instance
       else:
           return self.dedicated_models[symbol]
   ```

3. **Memory Budget Enforcement**:
   - Target: < 2GB total for all models
   - BTC base model: ~500MB
   - Light fine-tuned: ~500MB each
   - Max 3 concurrent model instances (1 base + 2 dedicated)

**Phase 4: Fine-Tuning Pipeline Generalization**

**Modify File**: `scripts/fine_tune_finbert.py`

**Make Asset-Agnostic**:

1. Add command-line arguments:
   ```
   python fine_tune_finbert.py \
       --symbol SOLUSDT \
       --base_model ./models/finbert-crypto-btc \
       --training_data ./data/fine_tuning/solana_headlines.jsonl \
       --output_path ./models/finbert-crypto-sol \
       --num_examples 100  # Light fine-tuning
   ```

2. **Light Fine-Tuning Mode** (for 0.7-0.9 correlation assets):
   - Use BTC fine-tuned model as starting point
   - Train on only 100 asset-specific examples
   - Freeze first 8 layers, only train last 4 layers
   - 1 epoch maximum (prevent overfitting)
   - Expected accuracy: 90-92% (vs 92-95% full fine-tuning)

3. **Cost Optimization**:
   - Full fine-tuning: 500 examples × $2.40 = $1,200
   - Light fine-tuning: 100 examples × $2.40 = $240
   - 5x cost reduction for correlated assets

**Phase 5: Gradual Rollout Strategy**

**Week 1-2: ETH Validation**:
1. Use BTC model on ETH data (no retraining)
2. Run CPCV validation on ETH historical data
3. Expected Sharpe: 90-95% of BTC Sharpe (due to high correlation)
4. If Sharpe < 80% of BTC, consider light fine-tuning

**Week 3-4: SOL Validation**:
1. Collect 100 Solana-specific headlines
2. Light fine-tune from BTC model
3. Run CPCV validation
4. Expected Sharpe: 85-90% of BTC Sharpe

**Week 5+: Altcoin Expansion**:
1. Analyze correlation for each new asset
2. Apply decision matrix (direct use vs light vs full fine-tuning)
3. Validate before production deployment

### Integration Points

**Modify File**: `primitives/integrated_signal_layer.py`

**Add Multi-Asset Support**:

1. Change initialization:
   ```
   def __init__(self, config, redis_client=None, asset_configs=None):
       self.asset_configs = asset_configs or ASSET_CONFIGS
       self.sentiment_manager = MultiAssetSentimentManager(self.asset_configs)
   ```

2. Modify `update()` signature:
   ```
   def update(self, symbol: str, ohlcv: Dict, ...):
       # Get asset-specific config
       asset_config = self.asset_configs.get(symbol)
       if not asset_config:
           logger.warning(f"No config for {symbol} - using default")

       # Get asset-specific model
       sentiment_analyzer = self.sentiment_manager.get_analyzer(symbol)

       # Use asset-specific lag horizons
       lag_features = self.lag_buffer.get_lag_features(
           symbol,
           horizons=asset_config.lag_horizons
       )
   ```

**Modify File**: `config.py`

**Add Multi-Asset Config Loading**:

```
def load_asset_configs() -> Dict[str, AssetConfig]:
    """Load asset-specific configurations from environment or defaults."""

    # Allow override via environment variable
    config_path = os.getenv('ASSET_CONFIG_PATH', './config/asset_configs.json')

    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    else:
        return ASSET_CONFIGS  # Use defaults
```

### Validation Criteria

**Per-Asset Validation**:

1. **ETH Validation** (no fine-tuning):
   - CPCV Sharpe ratio: > 2.8 (90% of BTC's 3.27)
   - Sentiment accuracy: > 85%
   - Latency: < 50ms p99

2. **SOL Validation** (light fine-tuning):
   - CPCV Sharpe ratio: > 2.5 (75% of BTC)
   - Sentiment accuracy: > 88%
   - Fine-tuning cost: < $300

3. **Memory Budget Test**:
   - Load models for BTC, ETH, SOL simultaneously
   - Measure total memory usage
   - Target: < 2GB total

4. **Correlation Tracking**:
   - Monitor sentiment correlation drift over time
   - If BTC-ETH correlation drops below 0.85, consider separate models
   - Quarterly reanalysis of correlation matrix

**Scalability Test**:

1. **50-Asset Load Test**:
   - Configure 50 crypto assets
   - Measure model loading time (target < 30 seconds)
   - Measure signal generation throughput (target > 100 signals/sec)

2. **Model Update Strategy**:
   - When BTC model is retrained, how to propagate to ETH/SOL?
   - Test model swap without downtime (load new model → switch pointer → unload old)

**Success Metrics**:
- ETH Sharpe > 2.8 with BTC model (no retraining)
- SOL Sharpe > 2.5 with light fine-tuning ($300 cost)
- Memory usage < 2GB for 5 concurrent assets
- Model loading time < 30 seconds
- Zero degradation in BTC performance when multi-asset enabled

---

## IMPLEMENTATION PRIORITY & TIMELINE

### Phase 1 (Week 1): Critical Infrastructure
**Priority: CRITICAL - Must complete before production**

1. **Enhancement 3: Latency Validation** (2 days)
   - Implement LatencyBenchmark class
   - Add SLA enforcement gates
   - Run benchmark suite
   - **Deliverable**: Validated p99 < 50ms

2. **Enhancement 6: Monitoring Setup** (3 days)
   - Implement metrics collector
   - Set up Prometheus + Grafana
   - Configure alerts
   - **Deliverable**: Operational dashboard with 6 panels

### Phase 2 (Week 2-3): High-Impact Enhancements
**Priority: HIGH - Significant Sharpe improvements**

3. **Enhancement 1: Sentiment Lags** (4 days)
   - Implement SentimentLagBuffer
   - Integrate with HybridSentimentAnalyzer
   - Run CPCV validation
   - **Deliverable**: +0.2 to +0.5 Sharpe improvement

4. **Enhancement 2: Dynamic Weighting** (5 days)
   - Implement DynamicSentimentWeighter
   - Define regime weight matrix
   - Integrate with regime detection
   - **Deliverable**: +21% accuracy improvement

### Phase 3 (Week 4-5): Model Quality
**Priority: HIGH - Long-term accuracy gains**

5. **Enhancement 4: Fine-Tuning** (7 days)
   - Collect 500 labeled examples ($1,200)
   - Train fine-tuned FinBERT
   - Validate accuracy > 92%
   - **Deliverable**: +15% accuracy, 2800% ROI

### Phase 4 (Week 6-8): Expansion
**Priority: MEDIUM - Scalability**

6. **Enhancement 5: Social Media** (10 days)
   - Set up Twitter/Reddit connectors
   - Implement spam filtering
   - Integrate Twitter-RoBERTa
   - **Deliverable**: +0.1 to +0.3 Sharpe improvement

7. **Enhancement 7: Multi-Asset** (5 days)
   - Implement MultiAssetSentimentManager
   - Validate on ETH (no retraining)
   - Validate on SOL (light fine-tuning)
   - **Deliverable**: ETH Sharpe > 2.8, SOL Sharpe > 2.5

### Total Timeline: 8 weeks

---

## RISK MITIGATION STRATEGIES

### Technical Risks

**Risk 1: Fine-tuning doesn't improve accuracy**
- **Probability**: LOW (research shows +15% gain)
- **Impact**: MEDIUM ($1,200 wasted)
- **Mitigation**:
  - Start with 100 examples pilot ($240)
  - Validate +5% improvement before scaling to 500
  - Fallback: Use baseline FinBERT

**Risk 2: Social media data too noisy**
- **Probability**: MEDIUM (70% spam rate common)
- **Impact**: MEDIUM (wasted compute + latency)
- **Mitigation**:
  - Implement strict spam filtering (target < 5% spam)
  - Run lead-lag analysis BEFORE full integration
  - Fallback: Disable social sentiment if correlation < 0.3

**Risk 3: Latency spikes in production**
- **Probability**: MEDIUM (FinBERT can be slow)
- **Impact**: HIGH (triggers SRM halts)
- **Mitigation**:
  - Enforce p99 < 50ms SLA gates
  - Implement circuit breaker (disable FinBERT if p99 > 100ms)
  - Consider model quantization (if doesn't increase latency)
  - Fallback: VADER-only mode (0.5ms latency)

**Risk 4: Model drift in production**
- **Probability**: MEDIUM (language evolves)
- **Impact**: HIGH (accuracy degradation)
- **Mitigation**:
  - Weekly manual validation (50 samples)
  - Accuracy drift alerts (< 80% → retrain)
  - Active learning pipeline (5 examples/day)
  - Quarterly retraining schedule

### Operational Risks

**Risk 5: Monitoring system downtime**
- **Probability**: LOW
- **Impact**: MEDIUM (blind to issues)
- **Mitigation**:
  - Prometheus HA mode (2 instances)
  - External uptime monitoring (UptimeRobot)
  - Fallback: Log-based monitoring

**Risk 6: Multi-asset correlation drift**
- **Probability**: MEDIUM (market conditions change)
- **Impact**: MEDIUM (wrong model used)
- **Mitigation**:
  - Quarterly correlation reanalysis
  - Alert if correlation drops > 0.1
  - Automatic fallback to light fine-tuning

### Financial Risks

**Risk 7: ROI below expectations**
- **Probability**: LOW
- **Impact**: LOW (still net positive)
- **Mitigation**:
  - Track Sharpe attribution per enhancement
  - Disable features with negative ROI
  - Quarterly ROI review

---

## SUCCESS CRITERIA & VALIDATION GATES

### Enhancement 1: Sentiment Lags
- ✅ Lag correlation > 0.7 at optimal horizon
- ✅ Sharpe improvement > +0.2
- ✅ No latency increase
- ✅ Memory usage < 5MB for 100 symbols

### Enhancement 2: Dynamic Weighting
- ✅ Accuracy improvement > +15%
- ✅ Sharpe improvement > +0.15
- ✅ Weight stability < 5 transitions/hour
- ✅ Regime classification accuracy > 90%

### Enhancement 3: Latency Validation
- ✅ p99 total latency < 50ms (idle)
- ✅ p99 total latency < 100ms (under load)
- ✅ Zero SLA breaches in 72-hour test
- ✅ Quantization validation framework operational

### Enhancement 4: Fine-Tuning
- ✅ Test set accuracy > 92%
- ✅ Sharpe improvement > +0.2
- ✅ ROI > 2000% in first year
- ✅ No catastrophic forgetting (general finance > 80%)

### Enhancement 5: Social Media
- ✅ Spam rate < 5% post-filtering
- ✅ Lead-lag correlation > 0.4
- ✅ Sharpe improvement > +0.1
- ✅ Twitter-RoBERTa accuracy > 85%

### Enhancement 6: Monitoring
- ✅ All 6 Grafana panels operational
- ✅ Alert firing latency < 5 minutes
- ✅ Zero false positives in 72-hour test
- ✅ Metric overhead < 1ms per update

### Enhancement 7: Multi-Asset
- ✅ ETH Sharpe > 2.8 (with BTC model)
- ✅ SOL Sharpe > 2.5 (with light fine-tuning)
- ✅ Memory usage < 2GB for 5 assets
- ✅ Model loading time < 30 seconds

### Overall System Validation
- ✅ Combined Sharpe improvement: +0.4 to +0.7 (cumulative across enhancements)
- ✅ Production uptime > 99.5% in shadow mode (72 hours)
- ✅ Zero SRM emergency halts due to latency
- ✅ Accuracy maintained > 85% over 30 days

---

## CONFIGURATION REFERENCE

### New Config Parameters to Add

**File: config.py - EnhancedSignalConfig additions**

```
# Enhancement 1: Sentiment Lags
sentiment_enable_lag_features: bool = True
sentiment_max_lag_bars: int = 360
sentiment_lag_horizons: Dict = field(default_factory=lambda: {
    'news': [30, 60, 90, 120, 180],
    'twitter': [15, 30, 45, 60, 120],
    'reddit': [180, 240, 270, 300, 360]
})
sentiment_bar_interval_minutes: int = 1

# Enhancement 2: Dynamic Weighting
sentiment_enable_dynamic_weighting: bool = True
sentiment_weight_smoothing_alpha: float = 0.1
sentiment_min_regime_duration: int = 5
sentiment_weight_change_limit: float = 0.15
volatility_thresholds: Dict = field(default_factory=lambda: {
    'low': 0.015, 'normal': 0.040, 'high': 0.040
})

# Enhancement 3: Latency Validation
enable_latency_gating: bool = True
latency_check_interval: int = 100
latency_slas: Dict = field(default_factory=lambda: {
    'sentiment_analysis': {'p99': 100, 'action': 'LOG_WARNING'},
    'total_signal': {'p99': 50, 'action': 'DISABLE_COMPONENT'}
})

# Enhancement 4: Fine-Tuning
sentiment_use_fine_tuned: bool = False
sentiment_fine_tuned_model_path: str = "./models/finbert-crypto-finetuned"

# Enhancement 5: Social Media
enable_social_sentiment: bool = False
twitter_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
social_spam_filter_enabled: bool = True
social_min_engagement: int = 5

# Enhancement 6: Monitoring
enable_prometheus_metrics: bool = True
prometheus_port: int = 8000
metrics_update_interval: int = 100

# Enhancement 7: Multi-Asset
asset_configs_path: str = "./config/asset_configs.json"
enable_multi_asset: bool = False
max_concurrent_models: int = 3
```

### Environment Variables

```
# Enhancement 4: Fine-Tuning
FINBERT_MODEL_PATH=./models/finbert-crypto-finetuned
USE_FINE_TUNED_MODEL=true

# Enhancement 5: Social Media
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Enhancement 6: Monitoring
PROMETHEUS_PORT=8000
GRAFANA_PORT=3000
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Enhancement 7: Multi-Asset
ASSET_CONFIG_PATH=./config/asset_configs.json
ENABLE_MULTI_ASSET=false
```

---

## TESTING STRATEGY

### Unit Tests (New Files to Create)

1. **tests/test_sentiment_lags.py**
   - Test buffer overflow handling
   - Test lag indexing accuracy
   - Test multi-symbol isolation
   - Test aggregation features

2. **tests/test_dynamic_weights.py**
   - Test regime classification
   - Test weight lookup
   - Test smoothing
   - Test fallback logic

3. **tests/test_latency_validator.py**
   - Test percentile calculation
   - Test SLA breach detection
   - Test circuit breaker
   - Test quantization validation

4. **tests/test_social_aggregator.py**
   - Test spam filtering
   - Test volume normalization
   - Test temporal windowing
   - Test aggregation methods

5. **tests/test_multi_asset_manager.py**
   - Test model pooling
   - Test asset routing
   - Test memory budget
   - Test model swapping

### Integration Tests

1. **tests/integration/test_end_to_end_with_lags.py**
   - Generate 500 bars of synthetic data
   - Verify lag features populated correctly
   - Verify fusion uses lag components
   - Verify output format unchanged

2. **tests/integration/test_regime_weighting.py**
   - Simulate regime transitions (Bull → Range → Bear)
   - Verify weights change appropriately
   - Verify smoothing prevents thrashing
   - Verify regime duration filter works

3. **tests/integration/test_social_pipeline.py**
   - Mock Twitter/Reddit APIs
   - Inject 1000 synthetic posts
   - Verify spam filtering removes 70%
   - Verify aggregated sentiment accurate

### CPCV Validation (Modify Existing)

**File: scripts/run_cpcv_validation.py**

**Add Ablation Studies**:

1. **Baseline**: Current system (no enhancements)
2. **+Lags**: Add sentiment lag features only
3. **+Lags+Dynamic**: Add dynamic weighting
4. **+Lags+Dynamic+FineTuned**: Add fine-tuned model
5. **+Lags+Dynamic+FineTuned+Social**: Add social media
6. **Full**: All enhancements enabled

**Expected Results**:
```
Configuration          Sharpe    Improvement    Status
----------------------------------------------------------
Baseline              3.27       -              [Baseline]
+Lags                 3.47       +0.20          [Target: +0.2]
+Lags+Dynamic         3.67       +0.40          [Target: +0.4]
+Lags+Dyn+Fine        3.87       +0.60          [Target: +0.6]
+Lags+Dyn+Fine+Social 4.00       +0.73          [Target: +0.7]
```

### Load Testing

**New File: scripts/load_test.py**

**Purpose**: Validate performance under production load

**Test Scenarios**:

1. **Sustained Load**: 1000 updates/second for 10 minutes
2. **Burst Load**: 5000 updates/second for 1 minute
3. **Multi-Asset Load**: 100 symbols × 10 updates/second

**Metrics to Track**:
- p99 latency (must stay < 100ms)
- Memory usage (must stay < 4GB)
- Error rate (must be < 0.1%)
- CPU usage (should be < 70%)

---

## ROLLBACK STRATEGY

### Scenario 1: Enhancement Degrades Performance

**Detection**: CPCV Sharpe < baseline OR latency > 2x baseline

**Rollback Procedure**:
1. Disable specific enhancement via feature flag
2. Restart IntegratedSignalLayer
3. Verify Sharpe returns to baseline
4. Investigate root cause offline

**Example**:
```python
# config.py
sentiment_enable_lag_features = False  # Rollback lags
sentiment_enable_dynamic_weighting = False  # Rollback dynamic weights
```

### Scenario 2: Production Accuracy Drift

**Detection**: `sentiment_accuracy_percent < 75` for 2 hours

**Rollback Procedure**:
1. Switch from fine-tuned model to baseline FinBERT
2. Alert ops team for model retraining
3. Continue operation with baseline (87% accuracy)

**Example**:
```python
# Environment variable override
USE_FINE_TUNED_MODEL=false
```

### Scenario 3: Latency SLA Breach

**Detection**: `p99 total_signal_latency > 100ms` for 5 minutes

**Automatic Rollback**:
1. Circuit breaker disables FinBERT (falls back to VADER-only)
2. Alert ops team
3. System continues at reduced accuracy but meeting latency SLA

**Example**:
```python
# Automatic in latency_validator.py
if p99_latency > 100:
    self.circuit_breaker.disable('finbert')
    logger.critical("FinBERT disabled due to latency breach")
```

### Scenario 4: Social Media Data Quality Issues

**Detection**: `spam_rate > 20%` OR `data_freshness > 600 seconds`

**Rollback Procedure**:
1. Disable social sentiment signals
2. Continue with news sentiment only
3. Investigate data pipeline

**Example**:
```python
# config.py
enable_social_sentiment = False
```

---

## DOCUMENTATION REQUIREMENTS

### For Each Enhancement, Create:

1. **Technical Specification Document**
   - Architecture diagram
   - Data flow diagram
   - API reference
   - Configuration reference

2. **Runbook**
   - Deployment procedure
   - Monitoring checklist
   - Troubleshooting guide
   - Rollback procedure

3. **Performance Benchmarks**
   - Latency measurements
   - Memory usage
   - Accuracy metrics
   - Sharpe improvements

4. **Code Comments**
   - Docstrings for all public methods
   - Inline comments for complex logic
   - Example usage in docstrings
   - Performance notes (e.g., "O(1) complexity")

---

## FINAL CHECKLIST BEFORE PRODUCTION

### Pre-Deployment Validation

- [ ] All 7 enhancements pass unit tests
- [ ] CPCV validation shows cumulative Sharpe > 3.6
- [ ] Load testing passes (1000 updates/sec, p99 < 100ms)
- [ ] Monitoring dashboard operational (all 6 panels)
- [ ] Alerting rules configured and tested
- [ ] Runbooks documented for all failure scenarios
- [ ] Rollback procedures tested
- [ ] Shadow mode run for 72 hours (zero incidents)

### Production Readiness

- [ ] Feature flags configured (gradual rollout plan)
- [ ] Circuit breakers tested (automatic fallbacks work)
- [ ] SRM integration validated (risk gating functional)
- [ ] Multi-asset configs validated (ETH, SOL tested)
- [ ] Fine-tuned model accuracy > 92%
- [ ] Social media spam filtering < 5%
- [ ] Latency SLAs enforced (p99 < 50ms)

### Operational Readiness

- [ ] Oncall team trained on runbooks
- [ ] Alert escalation policies configured
- [ ] Weekly manual validation scheduled
- [ ] Quarterly model retraining scheduled
- [ ] Monitoring dashboard shared with stakeholders
- [ ] Post-deployment review scheduled (30 days)

---

## EXPECTED OUTCOMES

### Performance Improvements

**Baseline (Current)**:
- Sharpe Ratio: 3.27
- Win Rate: 51.7%
- Max Drawdown: 43.2%
- Sentiment Accuracy: 87-89%
- Latency: 0.21ms average

**Target (All Enhancements)**:
- Sharpe Ratio: 3.6-3.9 (+10-19%)
- Win Rate: 53-55%
- Max Drawdown: 35-40%
- Sentiment Accuracy: 92-95%
- Latency: p99 < 50ms (validated)

### Cost-Benefit Analysis

**Investment**:
- Engineering time: ~320 hours (~8 weeks × 40 hours)
- Fine-tuning cost: $1,200 (one-time)
- Social media APIs: $100/month
- Monitoring infrastructure: $50/month
- **Total first-year cost**: ~$3,000

**Expected Benefit** (on $100K portfolio):
- Sharpe improvement +0.6 → Annual alpha +$18,000 to +$42,000
- **ROI**: 600% to 1400% annually

### Risk Reduction

- **Latency SLA enforcement** → Prevents SRM emergency halts
- **Accuracy drift monitoring** → Catches model degradation early
- **Circuit breakers** → Automatic fallbacks prevent silent failures
- **Multi-asset validation** → Scalable architecture for future growth

---

## CONCLUSION

This comprehensive guide provides antigravity coding agent with:

1. **7 complete enhancement specifications** (no code, just clear requirements)
2. **Validation criteria** for each enhancement
3. **Risk mitigation strategies** for common failure modes
4. **Rollback procedures** for production incidents
5. **Success metrics** with clear targets
6. **Implementation priority** and timeline (8 weeks)

**Key Principles**:
- ✅ Build incrementally (validate each enhancement before moving to next)
- ✅ Measure everything (if not measured, it didn't happen)
- ✅ Fail safely (circuit breakers, fallbacks, monitoring)
- ✅ Optimize ROI (focus on high-impact, low-complexity first)

**Next Steps for Antigravity Agent**:
1. Start with Enhancement 3 (Latency Validation) - CRITICAL for production
2. Follow with Enhancement 6 (Monitoring) - CRITICAL for visibility
3. Implement Enhancements 1+2 (Lags + Dynamic Weights) - HIGH impact
4. Complete Enhancements 4+5 (Fine-Tuning + Social) - MEDIUM-HIGH impact
5. Finally Enhancement 7 (Multi-Asset) - FUTURE scalability

**Estimated Timeline**: 8 weeks to full deployment with all enhancements validated and production-ready.

---

**Document Version**: 1.0
**Generated**: 2024-12-24
**Status**: READY FOR IMPLEMENTATION
**Author**: HIMARI Enhanced Layer 1 Team
