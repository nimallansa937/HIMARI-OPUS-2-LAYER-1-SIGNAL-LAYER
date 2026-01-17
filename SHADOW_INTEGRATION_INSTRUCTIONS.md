# Layer 1 Explorer → Hinance Shadow Integration Instructions

## **Step-by-Step Integration Guide**

This document explains how to connect Layer 1 Explorer to Hinance as the shadow environment.

---

## **Changes Required in main.py**

### **1. Add Import**

At the top of `main.py`, add:

```python
# NEW: Import Hinance client
from src.deployment.hinance_client import HinanceShadowEnvironment
```

### **2. Replace Mock Shadow Environment**

Find this line (around line 144):

```python
# OLD:
self.shadow = ShadowEnvironment()
```

Replace with:

```python
# NEW: Use Hinance instead of mock
if config.shadow_environment.provider == 'hinance':
    self.shadow = HinanceShadowEnvironment(
        hinance_url=config.shadow_environment.hinance_url,
        api_key=config.shadow_environment.hinance_api_key
    )
    logger.info(f"✅ Using Hinance shadow environment: {config.shadow_environment.hinance_url}")
else:
    # Fallback to mock for testing
    self.shadow = ShadowEnvironment()
    logger.info("⚠️ Using MOCK shadow environment (not real data)")
```

### **3. Update Config Loader**

In `src/infrastructure/config.py`, add shadow_environment config class:

```python
@dataclass
class ShadowEnvironmentConfig:
    """Shadow environment configuration."""
    provider: str = 'mock'  # 'mock' or 'hinance'
    hinance_url: str = 'http://localhost:8000'
    hinance_api_key: Optional[str] = None
    auto_deploy: bool = True
    min_backtest_sharpe: float = 1.5
    shadow_duration_days: int = 21
    shadow_capital_per_strategy: float = 5000
    check_interval_hours: int = 6
    transfer_ratio_threshold: float = 0.70
    max_drawdown_threshold: float = 0.15
    min_trade_count: int = 20
```

And add to Layer1ExplorerConfig:

```python
@dataclass
class Layer1ExplorerConfig:
    # ... existing fields ...

    # NEW: Add this field
    shadow_environment: ShadowEnvironmentConfig = field(default_factory=ShadowEnvironmentConfig)
```

---

## **Environment Variables**

Create `.env` file in Layer 1 root directory:

```bash
# Hinance Shadow Environment
HINANCE_API_KEY=your_api_key_here  # Optional, can be empty for local testing

# Leave other existing variables as-is
```

---

## **Testing the Integration**

### **Step 1: Start Hinance**

```bash
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\HINNANCE PAPER TRADING\hinance"

# Start Hinance with shadow integration
python src/main_shadow_integration.py
```

Check logs for:
```
✅ Shadow Bridge initialized
✅ Signal router initialized
🚀 Enhanced HINANCE Engine with Shadow Integration...
```

### **Step 2: Verify Hinance API**

Test the API:

```bash
# Check health
curl http://localhost:8000/health

# Should return: {"status":"healthy",...}

# List strategies (should be empty initially)
curl http://localhost:8000/shadow/strategies

# Should return: {"strategies":[],"total_count":0,...}
```

### **Step 3: Start Layer 1 Explorer**

```bash
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 1  EXPLORER AGENT"

# Run single cycle test
python main.py --single-cycle
```

Watch logs for:
```
✅ Using Hinance shadow environment: http://localhost:8000
Deploying strategy XXX to Hinance shadow...
✅ Strategy XXX deployed to Hinance. Allocated: $5000.00
```

### **Step 4: Monitor Shadow Trading**

While Layer 1 is running shadow, check Hinance:

```bash
# List active strategies
curl http://localhost:8000/shadow/strategies

# Get strategy performance
curl http://localhost:8000/shadow/strategies/{strategy_id}/performance
```

---

## **Full Integration Flow**

```
┌──────────────────────────────────┐
│ 1. Layer 1 generates strategy    │
│    via evolutionary/LLM engines  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 2. HIFA validation pipeline      │
│    - Backtest Sharpe > 1.5       │
│    - Passes all gates            │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 3. Layer 1: deploy to shadow     │
│    HinanceClient.run_shadow()    │
└────────────┬─────────────────────┘
             │
             │ HTTP POST
             ▼
┌──────────────────────────────────┐
│ 4. Hinance receives strategy     │
│    ShadowBridge.deploy_strategy()│
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 5. Signal Router starts          │
│    - WebSocket → Binance data    │
│    - Features extracted (60-dim) │
│    - Strategy evaluated on data  │
│    - Signals executed            │
└────────────┬─────────────────────┘
             │
             │ 21 days
             ▼
┌──────────────────────────────────┐
│ 6. Layer 1 polls performance     │
│    Every 6 hours via HTTP GET    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 7. Hinance returns metrics       │
│    - Sharpe ratio                │
│    - Transfer ratio (KEY!)       │
│    - Max drawdown                │
│    - Trade count, win rate, etc. │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 8. Layer 1 deployment decision   │
│    IF transfer_ratio >= 0.70 AND │
│       max_drawdown <= 0.15       │
│    THEN approve for live         │
│    ELSE reject                   │
└──────────────────────────────────┘
```

---

## **Verification Checklist**

After integration, verify:

- [ ] Hinance starts without errors
- [ ] Shadow Bridge API endpoints respond
- [ ] Layer 1 can reach Hinance (health check)
- [ ] Strategy deploys successfully
- [ ] Signal router evaluates strategy
- [ ] Performance metrics calculate correctly
- [ ] Transfer ratio matches expected value
- [ ] Layer 1 receives performance data
- [ ] Deployment decision logic works

---

## **Troubleshooting**

### **Error: "Failed to deploy strategy"**

**Cause**: Hinance not running or wrong URL

**Fix**:
```bash
# Check if Hinance is running
curl http://localhost:8000/health

# If not running, start it
cd hinance && python src/main_shadow_integration.py
```

### **Error: "Connection refused"**

**Cause**: Port 8000 already in use or firewall

**Fix**:
```bash
# Check what's on port 8000
netstat -ano | findstr :8000

# Or use different port in config
hinance_url: http://localhost:8001
```

### **Error: "Feature extraction failed"**

**Cause**: Not enough market data yet

**Fix**: Wait for WebSocket to receive at least 100 candles (~2 hours at 1min intervals)

### **Warn: "Using MOCK shadow environment"**

**Cause**: Config provider not set to 'hinance'

**Fix**: In `layer1.yaml`, ensure:
```yaml
shadow_environment:
  provider: hinance  # NOT 'mock'
```

---

## **Performance Expectations**

After full integration:

- **Latency**: Signal evaluation → execution < 100ms
- **Throughput**: 50 concurrent strategies @ 1000 updates/sec
- **Accuracy**: Transfer ratio ±5% of manual calculation
- **Uptime**: 99.9% with auto-reconnect

---

## **Next Steps After Integration**

1. **Run backtest validation**:
   - Compare shadow Sharpe vs backtest Sharpe
   - Verify transfer ratio calculation

2. **Load testing**:
   - Deploy 10 strategies
   - Monitor system resources
   - Scale to 50 strategies

3. **Live deployment**:
   - Wait for strategy to pass 21 days shadow
   - Verify transfer ratio > 0.70
   - Deploy to live trading (outside scope)

---

## **Support**

For issues:
1. Check Hinance logs: `hinance/logs/shadow_bridge.log`
2. Check Layer 1 logs: stdout
3. Verify database migrations ran: `psql -d hinance -c "\dt shadow*"`
4. Review integration plan: `LAYER1_SHADOW_INTEGRATION_PLAN.md`

---

**Integration Status**: Ready for testing
**Last Updated**: 2026-01-16
