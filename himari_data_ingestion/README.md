# HIMARI Data Ingestion Layer

**The Missing Link:** Connects exchange WebSockets to your Redpanda pipeline.

## The Problem This Solves

Your current infrastructure has:
```
[??? NOTHING ???] → Redpanda → Flink → Redis → Signal Layer
```

This package provides:
```
Binance WebSocket ─┐
Kraken WebSocket ──┼─→ DataIngestionService → Redpanda → Flink → Redis
CoinGecko REST ────┘
```

## Quick Start

```bash
# 1. Add to your data infrastructure
cd HIMARI-OPUS-DATA-INFRASTRUCTURE-
git clone [this-repo] src/ingestion

# 2. Install dependencies
pip install -r src/ingestion/requirements.txt

# 3. Configure API keys (optional for public endpoints)
cp src/ingestion/.env.example src/ingestion/.env

# 4. Start ingestion
python src/ingestion/main.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer (NEW)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ BinanceConnector │  │ KrakenConnector  │  │ CoinGeckoPoller  │   │
│  │   (WebSocket)    │  │   (WebSocket)    │  │     (REST)       │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                     │                     │              │
│           └──────────┬──────────┴──────────┬──────────┘              │
│                      │                     │                         │
│                      ▼                     ▼                         │
│           ┌─────────────────────────────────────────┐               │
│           │         Message Normalizer              │               │
│           │   (Converts to HIMARI OHLCV format)     │               │
│           └─────────────────────┬───────────────────┘               │
│                                 │                                    │
│                                 ▼                                    │
│           ┌─────────────────────────────────────────┐               │
│           │         Kafka Producer                  │               │
│           │   (Publishes to 'raw_market_data')      │               │
│           └─────────────────────┬───────────────────┘               │
│                                 │                                    │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Your Existing Infrastructure                        │
├─────────────────────────────────────────────────────────────────────┤
│   Redpanda (raw_market_data) → Flink → quality_scores → L1 Signals  │
└─────────────────────────────────────────────────────────────────────┘
```

## Supported Data Sources

### Tier 1: Real-Time WebSocket (Primary)
| Source | Status | Latency | Cost |
|--------|--------|---------|------|
| Binance | ✅ Implemented | <5ms | Free |
| Kraken | ✅ Implemented | <10ms | Free |
| Bybit | 🔲 Planned | <5ms | Free |
| Coinbase | 🔲 Planned | <10ms | Free |

### Tier 2: REST Polling (Backup/Historical)
| Source | Status | Latency | Cost |
|--------|--------|---------|------|
| CoinGecko | ✅ Implemented | 1-5s | Free (10K/month) |
| CoinCap | 🔲 Planned | 1-5s | Free |
| CoinMarketCap | 🔲 Planned | 1-5s | $29/month |

### Tier 3: Tick-Level (Premium)
| Source | Status | Latency | Cost |
|--------|--------|---------|------|
| Polygon.io | 🔲 Planned | <20ms | $49/month |

## Message Format

All connectors normalize data to this format before publishing:

```json
{
    "symbol": "BTCUSDT",
    "exchange": "binance",
    "timestamp": 1703289600000,
    "open": 43250.50,
    "high": 43312.00,
    "low": 43198.25,
    "close": 43275.00,
    "volume": 1234.567,
    "quote_volume": 53456789.12,
    "trades": 15234,
    "source": "websocket",
    "received_at": 1703289600005
}
```

## Configuration

```python
# config.py
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Kafka/Redpanda
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC_RAW = "raw_market_data"

# Exchange settings
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
KRAKEN_WS_URL = "wss://ws.kraken.com"

# Polling intervals
COINGECKO_POLL_INTERVAL = 60  # seconds
```

## Module Structure

```
src/ingestion/
├── __init__.py
├── main.py                    # Entry point
├── config.py                  # Configuration
├── requirements.txt
│
├── connectors/                # Exchange connectors
│   ├── __init__.py
│   ├── base.py               # Abstract base class
│   ├── binance.py            # Binance WebSocket
│   ├── kraken.py             # Kraken WebSocket
│   └── coingecko.py          # CoinGecko REST poller
│
├── normalizers/               # Message format converters
│   ├── __init__.py
│   └── ohlcv.py              # Normalize to HIMARI format
│
├── publishers/                # Kafka producers
│   ├── __init__.py
│   └── kafka_publisher.py    # Publish to Redpanda
│
└── utils/
    ├── __init__.py
    ├── healthcheck.py        # Connection monitoring
    └── metrics.py            # Prometheus metrics
```

## Monitoring

The ingestion layer exposes Prometheus metrics:

```
# Messages published
ingestion_messages_total{exchange="binance",symbol="BTCUSDT"} 12345

# Latency histogram
ingestion_latency_seconds_bucket{exchange="binance",le="0.001"} 10234

# Connection status
ingestion_connection_status{exchange="binance"} 1
```

## Error Handling

- **WebSocket disconnection:** Auto-reconnect with exponential backoff
- **Rate limiting:** Respects exchange limits, queues excess
- **Invalid data:** Logs warning, skips message, continues
- **Kafka unavailable:** Buffers locally, retries with backoff
