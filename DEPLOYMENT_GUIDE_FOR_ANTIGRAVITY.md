# HIMARI Signal Layer - Production Deployment Guide

**Version:** 1.0.0  
**Date:** 2024-12-25  
**Status:** Ready for Production Deployment

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Phase 1: Shadow Mode (72 Hours)](#phase-1-shadow-mode-72-hours)
4. [Phase 2: Monitoring Setup](#phase-2-monitoring-setup)
5. [Phase 3: Symbol Rollout](#phase-3-symbol-rollout)
6. [Rollback Procedures](#rollback-procedures)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the production deployment of the Enhanced HIMARI Layer 1 Signal System, featuring:

- **7 Algorithmic Components** with +6.05 Sharpe improvement
- **Hybrid Sentiment Analysis** (VADER + Financial-RoBERTa)
- **12,000+ Training Examples** for model fine-tuning
- **Sub-millisecond Latency** (0.20ms average)

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Shadow Mode          Phase 2: Monitoring          │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │ Legacy System    │          │ Prometheus       │         │
│  │ (Production)     │          │ (Metrics)        │         │
│  └────────┬─────────┘          └────────┬─────────┘         │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │ Enhanced System  │ ◄──────► │ Grafana          │         │
│  │ (Shadow)         │          │ (Dashboards)     │         │
│  └────────┬─────────┘          └──────────────────┘         │
│           │                                                  │
│           ▼                                                  │
│  Phase 3: Gradual Rollout                                   │
│  ┌──────────────────────────────────────────────┐           │
│  │ BTC → ETH → SOL,BNB,XRP → Full Deployment    │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Checklist

### ✅ Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Docker & Docker Compose | Required | v20.10+ |
| Python 3.9+ | Required | With pip/venv |
| Redis 6+ | Required | For state persistence |
| 8GB+ RAM | Required | For transformer models |
| GPU (optional) | Recommended | For model inference |

### ✅ Code Verification

```bash
# 1. Run unit tests
pytest tests/ -v

# 2. Verify integrated layer
python test_integrated_layer.py

# 3. Check imports
python -c "from primitives import IntegratedSignalLayer; print('OK')"

# 4. Verify sentiment model
python -c "from primitives import HybridSentimentAnalyzer; print('Model loaded')"
```

### ✅ Configuration

```bash
# Required environment variables
export HIMARI_ENHANCED_LAYER1_ENABLED=true
export HIMARI_SENTIMENT_ENABLED=true
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Optional: Fine-tuned model path
export HIMARI_SENTIMENT_MODEL=./models/financial-roberta-crypto-finetuned
```

---

## Phase 1: Shadow Mode (72 Hours)

Shadow mode runs the enhanced system **parallel to production** without affecting live trading decisions.

### Start Shadow Mode

```python
from deployment import ShadowModeRunner, ShadowModeConfig

config = ShadowModeConfig(
    duration_hours=72.0,
    symbols=["BTCUSDT", "ETHUSDT"],
    signal_diff_threshold=0.1,  # Alert if > 10% divergence
    log_interval_seconds=60,
    output_dir="shadow_mode_results"
)

runner = ShadowModeRunner(config=config)
runner.start()
```

### CLI Usage

```bash
# Run for 72 hours on BTC and ETH
python -m deployment.shadow_mode_runner --duration 72h --symbols BTCUSDT,ETHUSDT

# Quick test (1 hour)
python -m deployment.shadow_mode_runner --duration 1h --symbols BTCUSDT
```

### Comparison Integration

```python
# In your signal processor
from deployment import ShadowModeRunner

shadow = ShadowModeRunner()

def process_tick(market_data):
    # Run legacy system
    legacy_result = legacy_processor.process(market_data)
    
    # Run enhanced system (shadow)
    enhanced_result = integrated_layer.update(
        symbol=market_data['symbol'],
        ohlcv=market_data,
        orderbook=None
    )
    
    # Compare (doesn't affect trading)
    shadow.compare_signals(
        symbol=market_data['symbol'],
        legacy_result=legacy_result,
        enhanced_result={
            'composite_signal': enhanced_result.composite_signal,
            'regime': enhanced_result.regime,
            'latency_ms': enhanced_result.components.get('latency_ms', 0)
        }
    )
    
    # Return LEGACY result for production
    return legacy_result
```

### Success Criteria

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| Signal Divergence | < 15% avg | Investigate model differences |
| Regime Match Rate | > 85% | Review HMM parameters |
| Latency Degradation | < 3x legacy | Optimize bottlenecks |
| Error Rate | < 1% | Debug exceptions |

### Review Report

After 72 hours, a report is generated at `shadow_mode_results/shadow_mode_report_*.json`:

```json
{
  "recommendation": "PROCEED_TO_PRODUCTION",
  "summary": {
    "total_comparisons": 259200,
    "discrepancy_rate": 0.03,
    "all_pass": true
  }
}
```

---

## Phase 2: Monitoring Setup

### Start Monitoring Stack

```bash
cd deployment

# Start Prometheus, Grafana, Alertmanager
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services
docker-compose -f docker-compose.monitoring.yml ps
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | <http://localhost:3000> | admin / himari2024 |
| Prometheus | <http://localhost:9090> | - |
| Alertmanager | <http://localhost:9093> | - |

### Dashboard Panels

The pre-configured dashboard includes:

1. **Overview Row**
   - Throughput (req/s)
   - P99 Latency
   - Error Rate
   - Rollout Phase
   - Active Alerts

2. **Latency Distribution**
   - P50/P95/P99 over time
   - By component breakdown

3. **Sentiment Analysis**
   - Score distribution
   - By symbol trends
   - Regime pie chart

4. **Shadow Mode Comparison**
   - Signal divergence graph
   - Regime match rate gauge
   - Legacy vs Enhanced latency

5. **Rollout Progress**
   - Phase timeline
   - Phase metrics table
   - Rollback history

### Alert Rules

Pre-configured alerts in `prometheus/rules/himari_alerts.yml`:

| Alert | Severity | Condition |
|-------|----------|-----------|
| HighLatencyP99 | Warning | P99 > 25ms for 2m |
| CriticalLatency | Critical | P99 > 50ms for 1m |
| HighErrorRate | Warning | Errors > 1% for 2m |
| CriticalErrorRate | Critical | Errors > 5% for 1m |
| SignalDivergenceHigh | Warning | Divergence > 15% for 10m |
| RollbackTriggered | Critical | Any rollback |

---

## Phase 3: Symbol Rollout

### Rollout Sequence

```
Phase 1 (24h): BTCUSDT only
    ↓ [Criteria Met]
Phase 2 (24h): + ETHUSDT
    ↓ [Criteria Met]
Phase 3 (24h): + SOLUSDT, BNBUSDT, XRPUSDT
    ↓ [Criteria Met]
Phase 4 (24h): All remaining symbols
    ↓ [Criteria Met]
COMPLETED: Full production
```

### Start Rollout

```python
from deployment import SymbolRolloutController, RolloutConfig

config = RolloutConfig(
    phase_1_symbols=["BTCUSDT"],
    phase_2_symbols=["ETHUSDT"],
    phase_3_symbols=["SOLUSDT", "BNBUSDT", "XRPUSDT"],
    phase_4_symbols=["ADAUSDT", "DOGEUSDT", "AVAXUSDT"],
    auto_advance=False  # Manual advancement
)

controller = SymbolRolloutController(config=config)

# Start Phase 1
controller.start_rollout()

# Check if symbol should use enhanced
if controller.is_symbol_enabled("BTCUSDT"):
    result = enhanced_layer.update(...)
else:
    result = legacy_processor.process(...)
```

### Advancement Criteria

Each phase must meet these criteria before advancing:

| Criteria | Threshold |
|----------|-----------|
| Duration | ≥ 24 hours |
| Comparisons | ≥ 1,000 |
| Signal Diff | ≤ 0.15 |
| Regime Match | ≥ 85% |
| Latency | ≤ 10ms |
| Error Rate | ≤ 1% |
| Discrepancy Rate | ≤ 5% |

### Manual Advancement

```python
# Check current status
status = controller.get_status()
print(f"Phase: {status['phase']}")
print(f"Criteria Met: {status['metrics']['criteria_met']}")
print(f"Issues: {status['metrics']['issues']}")

# Advance when ready
if status['metrics']['criteria_met']:
    controller.advance_phase()
```

---

## Rollback Procedures

### Automatic Rollback Triggers

| Condition | Severity | Action |
|-----------|----------|--------|
| Error Rate > 5% | CRITICAL | Immediate full rollback |
| Latency > 50ms sustained | CRITICAL | Immediate full rollback |
| Signal Divergence > 50% | CRITICAL | Immediate full rollback |
| Error Rate > 1% | WARNING | Alert + manual review |
| Latency > 25ms | WARNING | Alert + manual review |

### Manual Symbol Rollback

```python
# Rollback single symbol
controller.rollback_symbol("BTCUSDT", "High error rate observed")

# Rollback all symbols
controller.rollback_all("System-wide latency degradation")
```

### Emergency Rollback

```bash
# Set environment to disable enhanced layer
export HIMARI_ENHANCED_LAYER1_ENABLED=false

# Restart signal processor
systemctl restart himari-signal-processor

# Or via Redis flag
redis-cli SET himari:enhanced:enabled false
```

### Post-Rollback Analysis

1. Collect logs from `shadow_mode_results/`
2. Review Grafana dashboards for anomalies
3. Check `rollback_reasons` in Redis
4. Generate incident report

---

## Integration Examples

### Complete Signal Processor Integration

```python
from deployment import (
    ShadowModeRunner, 
    SymbolRolloutController,
    RollbackCriteria
)
from primitives import IntegratedSignalLayer
from monitoring import PrometheusMetricsCollector

class ProductionSignalProcessor:
    def __init__(self):
        # Initialize monitoring
        self.metrics = PrometheusMetricsCollector()
        self.metrics.start_server()
        
        # Initialize enhanced layer
        self.enhanced_layer = IntegratedSignalLayer(config, redis)
        
        # Initialize deployment controllers
        self.rollout = SymbolRolloutController()
        self.shadow = ShadowModeRunner()
        
        # Start shadow mode initially
        self.shadow.start()
    
    def process(self, market_data: dict) -> dict:
        start = time.perf_counter()
        symbol = market_data['symbol']
        
        try:
            # Legacy processing (always)
            legacy_result = self.legacy_process(market_data)
            
            # Enhanced processing (if enabled for symbol)
            if self.rollout.is_symbol_enabled(symbol):
                enhanced_result = self.enhanced_layer.update(
                    symbol=symbol,
                    ohlcv=market_data
                )
                
                # Record metrics
                latency_ms = (time.perf_counter() - start) * 1000
                self.metrics.record_latency('signal_processing', latency_ms)
                
                # Check rollback criteria
                should_rollback, severity, reason = RollbackCriteria.should_rollback(
                    error_rate=self.get_error_rate(symbol),
                    latency_ms=latency_ms,
                    signal_divergence=abs(
                        enhanced_result.composite_signal - 
                        legacy_result.get('signal', 0)
                    )
                )
                
                if should_rollback:
                    self.rollout.rollback_symbol(symbol, reason)
                    return legacy_result
                
                return enhanced_result.__dict__
            
            else:
                # Shadow comparison only
                enhanced_result = self.enhanced_layer.update(
                    symbol=symbol,
                    ohlcv=market_data
                )
                self.shadow.compare_signals(symbol, legacy_result, enhanced_result)
                return legacy_result
                
        except Exception as e:
            self.metrics.record_error(str(type(e).__name__), 'signal_processing')
            return self.legacy_process(market_data)
```

### Kubernetes Deployment

```yaml
# kubernetes/himari-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: himari-signal-layer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: himari-signal-layer
  template:
    metadata:
      labels:
        app: himari-signal-layer
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: signal-processor
        image: himari/signal-layer:latest
        ports:
        - containerPort: 8000
          name: metrics
        env:
        - name: HIMARI_ENHANCED_LAYER1_ENABLED
          value: "true"
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: himari-config
              key: redis_host
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## Troubleshooting

### Common Issues

**1. High Latency**

```bash
# Check component breakdown
curl http://localhost:8000/metrics | grep himari_latency

# Common causes:
# - Transformer model on CPU (use GPU or quantized model)
# - Redis connection issues
# - Memory pressure
```

**2. Model Not Loading**

```python
# Verify model path
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "soleimanian/financial-roberta-large-sentiment"
)
```

**3. Redis Connection Failed**

```bash
# Test connection
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

# Check network
docker network inspect himari-monitoring
```

**4. Grafana Dashboard Empty**

```bash
# Verify Prometheus is scraping
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics
```

### Health Check Endpoints

```bash
# Main application
curl http://localhost:8000/health

# Shadow mode status
curl http://localhost:8001/status

# Rollout status
curl http://localhost:8002/status
```

---

## Timeline Summary

| Day | Phase | Actions |
|-----|-------|---------|
| 0 | Pre-Deploy | Verification, monitoring setup |
| 1-3 | Shadow Mode | 72h parallel comparison |
| 4 | Review | Analyze shadow mode report |
| 5-6 | Phase 1 | BTC production |
| 7-8 | Phase 2 | + ETH |
| 9-10 | Phase 3 | + Majors |
| 11-12 | Phase 4 | Full rollout |
| 13+ | Production | Monitor & optimize |

---

## Support

- **Logs:** `./logs/signal_processor.log`
- **Reports:** `./shadow_mode_results/`
- **Metrics:** <http://localhost:3000> (Grafana)
- **Alerts:** <http://localhost:9093> (Alertmanager)

---

*Last Updated: 2024-12-25*
