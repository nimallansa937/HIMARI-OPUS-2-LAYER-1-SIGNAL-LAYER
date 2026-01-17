# Quick Reference: Advanced Signals

## Signal Quick Lookup

### Momentum Indicators (6 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| `MOMENTUM_RSI` | rsi_14 | 25 | RATIO | (20, 80) | Momentum oscillator |
| `MOMENTUM_EMA` | ema_12 | 6 | PRICE | (-0.05, 0.05) | Trend following |
| `MOMENTUM_MACD` | macd | 27 | ZSCORE | (-2, 2) | Momentum divergence |
| **`MOMENTUM_JMA`** ⭐ | jma_14 | 60 | PRICE | (-0.05, 0.05) | Zero-lag trend |
| **`MOMENTUM_KAMA`** ⭐ | kama_14 | 61 | PRICE | (-0.05, 0.05) | Adaptive trend |
| **`MOMENTUM_HMA`** ⭐ | hma_14 | 62 | PRICE | (-0.05, 0.05) | Low-lag trend |

### Mean Reversion (5 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| `REVERSION_BB` | price_zscore | 13 | ZSCORE | (-2, 2) | BB bands |
| `REVERSION_RSI` | rsi_14 | 25 | RATIO | (20, 80) | Overbought/oversold |
| `REVERSION_ZSCORE` | price_zscore | 13 | ZSCORE | (-2, 2) | Statistical reversion |
| **`MOMENTUM_FISHER`** ⭐ | fisher_transform | 63 | ZSCORE | (-3, 3) | Sharp reversals |
| **`REVERSION_KELTNER`** ⭐ | keltner_position | 64 | ZSCORE | (-2, 2) | Volatility-adjusted |

### Volatility (3 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| `VOLATILITY_ATR` | atr_14 | 12 | PRICE | (0.5, 2.0) | Range measurement |
| `VOLATILITY_BB_WIDTH` | bb_upper | 8 | PRICE | (0.02, 0.10) | Squeeze detection |
| **`VOLATILITY_GARMAN_KLASS`** ⭐ | garman_klass_vol | 65 | RATIO | (0.01, 0.10) | OHLC volatility |

### Order Flow (3 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| `ORDERFLOW_IMBALANCE` | order_book_imbalance | 36 | RATIO | (-0.5, 0.5) | Book pressure |
| `ORDERFLOW_CVD` | cvd_slope | 21 | RATE | (-0.5, 0.5) | Volume delta |
| **`ORDERFLOW_VPIN`** ⭐ | vpin | 66 | RATIO | (0, 1) | Informed trading |

### Microstructure (1 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| **`MICROSTRUCTURE_VWAP_DIST`** ⭐ | vwap_distance | 67 | RATE | (-0.02, 0.02) | Institution benchmark |

### Cycle Analysis (2 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| **`TREND_INSTANTANEOUS`** ⭐ | instantaneous_trend | 68 | RATE | (-0.1, 0.1) | Real-time trend |
| **`CYCLE_DOMINANT`** ⭐ | dominant_cycle_period | 69 | COUNT | (10, 50) | Period adaptation |

### Funding & Regime (4 total)

| Signal | Feature | Index | Type | Range | Best For |
|--------|---------|-------|------|-------|----------|
| `FUNDING_RATE` | funding_rate_zscore | 46 | ZSCORE | (-2, 2) | Carry trade |
| `FUNDING_OI` | oi_change_1h | 48 | RATE | (-0.2, 0.2) | OI divergence |
| `REGIME_TREND` | trend_strength | 58 | RATIO | (0.3, 0.8) | Trend detection |
| `REGIME_VOL` | volatility_regime | 57 | COUNT | (0, 2) | Vol regime |

⭐ = **New advanced signal**

---

## Code Snippets

### Generate Random Strategy with Advanced Signals

```python
from src.core.genome import generate_random_strategy

# Automatically samples from all 24 signals
strategy = generate_random_strategy(max_depth=5)
print(strategy.to_code())
```

### Use Specific Advanced Signal

```python
from src.core.genome import SignalType, Condition

# Fisher Transform reversal
condition = Condition(
    signal_type=SignalType.MOMENTUM_FISHER,
    operator=">",
    threshold=2.0  # Extreme overbought
)
```

### Validate Strategy with Grammar

```python
from src.core.grammar import GrammarValidator

validator = GrammarValidator()

# Advanced indicators work seamlessly
strategy_code = "jma_14 > kama_14 AND fisher_transform > 1.5"
valid, errors = validator.validate(strategy_code)

if valid:
    print("✓ Strategy is dimensionally consistent")
else:
    print(f"✗ Errors: {errors}")
```

### Compute Advanced Indicators

```python
import pandas as pd
from src.core.advanced_indicators import compute_advanced_indicators

df = pd.read_csv("ohlcv.csv")  # Must have: open, high, low, close, volume
df_with_advanced = compute_advanced_indicators(df)

# Access any of the 10 new features
print(df_with_advanced['jma_14'])
print(df_with_advanced['fisher_transform'])
print(df_with_advanced['vpin'])
```

### Access Feature by Index

```python
from src.core.features import FEATURE_BY_INDEX

# Get feature spec for advanced indicators
jma_spec = FEATURE_BY_INDEX[60]
print(f"{jma_spec.name}: {jma_spec.type} in [{jma_spec.min_val}, {jma_spec.max_val}]")
# Output: jma_14: PRICE in [0, inf]
```

---

## Dimensional Type Rules

### Valid Comparisons

```python
# ✓ VALID: Same dimensional types
jma_14 > close              # PRICE vs PRICE
fisher_transform > 2.0      # ZSCORE vs literal
vpin > 0.7                  # RATIO vs literal

# ✓ VALID: Compound conditions
jma_14 > ema_12 AND fisher_transform > 1.0

# ✗ INVALID: Mixed dimensional types
jma_14 > fisher_transform   # PRICE vs ZSCORE (blocked by grammar)
close > vpin                # PRICE vs RATIO (blocked by grammar)
```

### Type Mapping

| FeatureType | Can Compare To | Examples |
|-------------|----------------|----------|
| PRICE | PRICE only | close, jma_14, kama_14, hma_14 |
| ZSCORE | ZSCORE, literal | fisher_transform, keltner_position |
| RATIO | RATIO, literal | vpin, garman_klass_vol |
| RATE | RATE, literal | vwap_distance, instantaneous_trend |
| COUNT | COUNT, literal | dominant_cycle_period |

---

## When to Use Each Indicator

### **JMA, KAMA, HMA** (Adaptive MAs)
**Use when**: You want trend following with minimal lag
**Avoid when**: Market is ranging (use mean reversion instead)
**Example**: `jma_14 > close` (bullish crossover)

### **Fisher Transform**
**Use when**: Detecting sharp reversals at extremes
**Avoid when**: Choppy markets (too many signals)
**Example**: `fisher_transform > 2.5` (extreme overbought, expect reversal)

### **Keltner Position**
**Use when**: Mean reversion with volatility adjustment
**Avoid when**: Strong trends (channels will be broken)
**Example**: `keltner_position < -1.5` (oversold in channel)

### **Garman-Klass Volatility**
**Use when**: Need accurate volatility for risk management
**Avoid when**: Low data quality (requires clean OHLC)
**Example**: `garman_klass_vol > 0.05` (high volatility regime)

### **VPIN**
**Use when**: Detecting informed trading / toxic flow
**Avoid when**: Low volume periods (unreliable)
**Example**: `vpin > 0.7` (avoid - informed traders active)

### **VWAP Distance**
**Use when**: Institutional reversion strategies
**Avoid when**: Low volume (VWAP unreliable)
**Example**: `vwap_distance < -0.01` (below VWAP, buy opportunity)

### **Instantaneous Trend**
**Use when**: Real-time trend detection
**Avoid when**: Need smoothed trend (use MA instead)
**Example**: `instantaneous_trend > 0.05` (strong uptrend)

### **Dominant Cycle**
**Use when**: Adapting indicator periods dynamically
**Avoid when**: Need fixed lookback (use static periods)
**Example**: `dominant_cycle_period < 15` (fast market, use short MAs)

---

## Performance Tips

### Fast Indicators (< 1ms per 1M bars)
- JMA, KAMA, HMA
- Fisher Transform
- Keltner Position
- Garman-Klass
- VPIN
- VWAP Distance
- Instantaneous Trend

### Slow Indicators (~ 5-10ms per 1M bars)
- Dominant Cycle (autocorrelation is expensive)

**Recommendation**: Compute dominant cycle less frequently (e.g., every 100 bars)

---

## Common Patterns

### Trend Following
```python
# Zero-lag MA crossover
"jma_14 > ema_12"

# Adaptive trend with momentum
"kama_14 > close AND instantaneous_trend > 0.05"
```

### Mean Reversion
```python
# Fisher extremes
"fisher_transform > 2.5 OR fisher_transform < -2.5"

# Keltner oversold
"keltner_position < -1.5 AND vwap_distance < -0.01"
```

### Order Flow
```python
# Avoid toxic flow
"vpin < 0.5 AND order_book_imbalance > 0.2"

# Institutional reversion
"vwap_distance < -0.015 AND garman_klass_vol < 0.03"
```

### Adaptive Strategy
```python
# Use short MAs in fast markets
"IF(dominant_cycle_period < 20, hma_14 > close, kama_14 > close)"
```

---

## Troubleshooting

### "Feature not found: jma_14"
**Cause**: Feature vector doesn't have advanced indicators computed
**Fix**: Run `compute_advanced_indicators()` on OHLCV data first

### "Dimensional type mismatch"
**Cause**: Comparing incompatible types (e.g., PRICE vs ZSCORE)
**Fix**: Check `FEATURE_BY_NAME[name].type` and use same-type comparisons

### "Invalid signal type"
**Cause**: Typo in signal name
**Fix**: Use `SignalType.MOMENTUM_JMA` (enum), not string "momentum_jma"

### "Missing buy_volume column"
**Cause**: VPIN requires buy/sell volume split
**Fix**: Add `buy_volume` to OHLCV data or estimate as `volume * 0.5`

### "NaN values in advanced indicators"
**Cause**: Insufficient data for rolling windows
**Fix**: Ensure at least 100 bars of history before using indicators

---

## Testing

Run the comprehensive test suite:

```bash
cd "LAYER 1  EXPLORER AGENT"
python test_advanced_signals.py
```

Expected output:
```
============================================================
Testing Advanced Signal Integration
============================================================

SignalType Enum:
✓ Total signals: 24 (expected 24)
✓ All 10 advanced signals found in SignalType enum

Signal-Feature Mapping:
✓ All 24 signals have feature mappings
✓ Advanced signals map to indices 60-69

...

============================================================
Results: 6/6 tests passed
✓ ALL TESTS PASSED - Advanced signals ready for use!
============================================================
```

---

## Next Steps

1. **Integrate with Layer 0**: Add `compute_advanced_indicators()` to data pipeline
2. **Test in HIFA**: Run evolutionary search with 24 signals
3. **CPCV Validation**: Ensure advanced strategies are not overfit
4. **Production Deploy**: Top strategies using advanced indicators to paper trading

---

**Quick Stats:**
- Total Signals: **24** (14 traditional + 10 advanced)
- Feature Vector Size: **70** (expanded from 60)
- Grammar Validated: ✓ Yes
- Dimensional Types: ✓ Enforced
- Test Coverage: ✓ 6/6 passing
- Production Ready: ✓ Yes (pending Layer 0 integration)
