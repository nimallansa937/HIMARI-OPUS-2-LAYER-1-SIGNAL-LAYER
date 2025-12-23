# HIMARI Layer 1 Signal Layer - Production Implementation

**Integrates with:** https://github.com/nimallansa937/HIMARI-OPUS-DATA-INFRASTRUCTURE-

This package implements the enhanced L1 Signal Layer on top of your existing data infrastructure (Redpanda → Flink → Redis).

## Quick Start

```bash
# 1. Clone into your existing HIMARI infrastructure
cd HIMARI-OPUS-DATA-INFRASTRUCTURE-
git clone [this-repo] src/layer1

# 2. Install additional dependencies
pip install -r src/layer1/requirements.txt

# 3. Run signal layer on top of existing Flink pipeline
python src/layer1/signal_processor.py
```

## Architecture Integration

```
Your Existing Infrastructure          New L1 Signal Layer
========================          =====================
                                  
Binance/Kraken WebSocket          
        ↓                         
   Redpanda ──────────────────→  SignalProcessor
        ↓                              ↓
   Flink Pipeline                 ┌────┴────┐
   (Quality Validation)           │ Engines │
        ↓                         ├─────────┤
   Redis Feature Store ←──────────│ Kalman  │
        ↓                         │ HMM     │
   TimescaleDB                    │ GARCH   │
                                  │ Hurst   │
                                  └────┬────┘
                                       ↓
                                  Redis signals:*
                                       ↓
                                  Layer 2 (Tactical)
```

## Module Structure

```
src/layer1/
├── __init__.py
├── requirements.txt
├── config.py                 # Configuration & Redis keys
├── signal_processor.py       # Main entry point (Kafka consumer)
│
├── primitives/               # O(1) streaming algorithms
│   ├── __init__.py
│   ├── welford.py           # Online variance
│   ├── kalman.py            # Kalman filter
│   ├── ultimate_smoother.py # Ehlers 2024
│   ├── rls.py               # Recursive least squares
│   └── tdigest_wrapper.py   # Streaming quantiles
│
├── regime/                   # Regime detection
│   ├── __init__.py
│   ├── hmm_forward.py       # Hidden Markov Model
│   ├── garch.py             # Online GARCH(1,1)
│   ├── hurst.py             # Moving Hurst exponent
│   └── entropy.py           # Sample entropy
│
├── signals/                  # Signal generators
│   ├── __init__.py
│   ├── momentum.py          # Momentum signals
│   ├── mean_reversion.py    # Mean reversion signals
│   ├── volatility.py        # Volatility signals
│   └── volume.py            # Volume/OBI signals
│
├── validation/               # Strategy validation
│   ├── __init__.py
│   ├── deflated_sharpe.py   # DSR with 3.0 hurdle
│   ├── cpcv.py              # Combinatorial purged CV
│   └── spa_test.py          # White/Hansen tests
│
└── tests/                    # Unit & integration tests
    ├── __init__.py
    ├── test_primitives.py
    ├── test_regime.py
    ├── test_signals.py
    └── test_integration.py
```

## Redis Key Schema

Signals are stored in Redis with this schema:

```
signals:{symbol}:latest          # Hash: all current signals
signals:{symbol}:momentum        # Float: -1 to +1
signals:{symbol}:mean_reversion  # Float: -1 to +1
signals:{symbol}:volatility      # Float: 0 to 1 (normalized)
signals:{symbol}:regime          # String: BULL|BEAR|RANGE
signals:{symbol}:regime_conf     # Float: 0 to 1 (confidence)
signals:{symbol}:obi             # Float: -1 to +1 (order book imbalance)
signals:{symbol}:hurst           # Float: 0 to 1
signals:{symbol}:entropy         # Float: 0 to 2
signals:{symbol}:timestamp       # Int: Unix ms
```

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Signal update latency | <10ms | TBD |
| Memory per symbol | <1MB | TBD |
| Throughput | >10,000 msg/sec | TBD |

## Testing

```bash
# Unit tests
pytest src/layer1/tests/ -v

# Integration test with Redis
pytest src/layer1/tests/test_integration.py -v

# Performance benchmark
python src/layer1/benchmark.py
```
