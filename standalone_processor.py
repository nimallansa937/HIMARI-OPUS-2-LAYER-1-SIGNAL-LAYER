"""
Standalone Signal Processor for HIMARI L1

This version uses absolute imports and can be run directly without
needing to set up the package structure.
"""

import json
import time
import logging
import signal
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import redis
from kafka import KafkaConsumer
import numpy as np

# Absolute imports (not relative)
from primitives.welford import WelfordVariance
from primitives.kalman import KalmanFilter
from primitives.ultimate_smoother import UltimateSmoother
from primitives.rls import RegressionChannel
from primitives.garch import OnlineGARCH
from primitives.hurst import MovingHurst
from primitives.volume import SyntheticVolumeDelta, RelativeVolume, OrderBookImbalance
from primitives.tdigest_quantiles import StreamingQuantiles

from ml.lorentzian_knn import LorentzianKNN
from ml.ensemble import EnsembleFusion

from fusion.dempster_shafer import DempsterShafer

from regime import StreamingHMM

from config import (
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_SSL, REDIS_DB,
    KAFKA_BOOTSTRAP, KAFKA_CONSUMER_GROUP, KAFKA_INPUT_TOPIC,
    RedisKeys, L1Config, DEFAULT_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('himari.l1')


@dataclass
class SymbolState:
    """Per-symbol processing state with all HIMARI L1 components."""
    symbol: str
    
    # Tier 5: Streaming Primitives
    kalman: KalmanFilter = field(default_factory=lambda: KalmanFilter(0.01, 0.1))
    smoother: UltimateSmoother = field(default_factory=lambda: UltimateSmoother(20))
    returns_welford: WelfordVariance = field(default_factory=WelfordVariance)
    garch: OnlineGARCH = field(default_factory=OnlineGARCH)
    volume_delta: SyntheticVolumeDelta = field(default_factory=SyntheticVolumeDelta)
    rvol: RelativeVolume = field(default_factory=RelativeVolume)
    obi: OrderBookImbalance = field(default_factory=OrderBookImbalance)
    tdigest: StreamingQuantiles = field(default_factory=StreamingQuantiles)
    
    # Tier 4: DSP/Regime
    hurst: MovingHurst = field(default_factory=lambda: MovingHurst(100, 10))
    hmm: StreamingHMM = field(default_factory=lambda: StreamingHMM(3))
    regression: RegressionChannel = field(default_factory=lambda: RegressionChannel(0.99, 2.0))
    
    # Tier 3: ML Prediction
    lorentzian: LorentzianKNN = field(default_factory=lambda: LorentzianKNN(k=20, feature_dim=15))
    ensemble: EnsembleFusion = field(default_factory=lambda: EnsembleFusion(
        num_models=4, model_names=['kalman', 'lorentzian', 'hmm', 'hurst']
    ))
    
    # Tier 1: Signal Fusion
    dempster_shafer: DempsterShafer = field(default_factory=DempsterShafer)
    
    # Tracking
    last_price: float = 0.0
    last_timestamp: int = 0
    tick_count: int = 0
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(50))
    
    def compute_return(self, price: float) -> float:
        if self.last_price <= 0:
            return 0.0
        return np.log(price / self.last_price)


class StandaloneSignalProcessor:
    """Standalone signal processor with absolute imports."""
    
    def __init__(self, config: Optional[L1Config] = None):
        self.config = config or DEFAULT_CONFIG
        self._running = False
        self._states: Dict[str, SymbolState] = {}
        
        # Initialize Redis
        self._redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            ssl=REDIS_SSL,
            db=REDIS_DB,
            decode_responses=True,
            socket_timeout=5.0,
        )
        
        # Initialize Kafka consumer
        self._consumer = KafkaConsumer(
            KAFKA_INPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_CONSUMER_GROUP,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        )
        
        # Metrics
        self._metrics = {
            'messages_processed': 0,
            'signals_published': 0,
            'errors': 0,
        }
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
        logger.info(f"SignalProcessor initialized for symbols: {self.config.symbols}")
    
    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol=symbol)
            logger.info(f"Initialized state for {symbol}")
        return self._states[symbol]
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process through 5-tier pipeline."""
        try:
            symbol = message.get('symbol', '').upper()
            price = float(message.get('price', 0))
            timestamp = int(message.get('timestamp', 0))
            volume = float(message.get('volume', 0))
            high = float(message.get('high', price))
            low = float(message.get('low', price))
            open_price = float(message.get('open', price))
            
            if not symbol or price <= 0:
                return None
            
            state = self._get_state(symbol)
            ret = state.compute_return(price)
            
            # === TIER 5: PRIMITIVES ===
            smoothed = state.smoother.update(price)
            kalman = state.kalman.update(price)
            if abs(ret) < 0.5:
                state.returns_welford.update(ret)
            garch_vol = state.garch.update(ret)
            state.tdigest.update(price)
            
            # === TIER 4: DSP/REGIME ===
            hurst_val, hurst_regime = state.hurst.update(price)
            hmm_state, hmm_conf = state.hmm.update(ret)
            # HMM has probabilities property, not method
            probs = state.hmm.probabilities
            regime_probs = {'BULL': probs[0], 'BEAR': probs[1], 'RANGE': probs[2] if len(probs) > 2 else 0.34}
            reg = state.regression.update(state.tick_count, price)
            
            # === TIER 2: VOLUME ===
            vol_delta = state.volume_delta.update(open_price, high, low, price, volume)
            rvol = state.rvol.update(volume, timestamp)
            obi = state.obi.update(open_price, high, low, price)
            
            # === TIER 3: ML ===
            features = np.array([ret, hurst_val, garch_vol, vol_delta, obi] + [0]*10)
            p_bull, knn_conf = state.lorentzian.predict(features)
            # EnsembleFusion expects list of (prediction, confidence) or fusions
            ensemble_sig = float(regime_probs['BULL'])
            ensemble_agree = hmm_conf
            
            # === TIER 1: FUSION ===
            state.dempster_shafer.reset()
            state.dempster_shafer.add_evidence({'bullish': p_bull, 'bearish': 1-p_bull})
            ds_decision, ds_belief, ds_uncertainty = state.dempster_shafer.get_decision()
            
            signals = {
                'symbol': symbol,
                'timestamp': timestamp,
                'price': price,
                'regime': ['BULL', 'BEAR', 'RANGE'][hmm_state],
                'regime_confidence': hmm_conf,
                'hurst': hurst_val,
                'p_bullish': p_bull,
                'ensemble_signal': ensemble_sig,
                'ds_decision': ds_decision,
                'garch_vol': garch_vol,
                'volume_delta': vol_delta,
                'obi': obi,
            }
            
            state.last_price = price
            state.last_timestamp = timestamp
            state.tick_count += 1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error: {e}")
            self._metrics['errors'] += 1
            return None
    
    def publish_signals(self, signals: Dict[str, Any]) -> None:
        """Publish to Redis."""
        symbol = signals['symbol']
        key = f"signals:{symbol}:latest"
        self._redis.hset(key, mapping={k: str(v) for k, v in signals.items()})
        self._metrics['signals_published'] += 1
    
    def run(self) -> None:
        """Main processing loop."""
        self._running = True
        logger.info("Starting signal processor... (Ctrl+C to stop)")
        
        try:
            while self._running:
                messages = self._consumer.poll(timeout_ms=100)
                
                for tp, records in messages.items():
                    for record in records:
                        signals = self.process_message(record.value)
                        if signals:
                            self.publish_signals(signals)
                            if self._metrics['signals_published'] % 100 == 0:
                                logger.info(f"Processed {self._metrics['signals_published']} signals")
                        self._metrics['messages_processed'] += 1
        finally:
            self._cleanup()
    
    def _shutdown(self, signum, frame):
        logger.info("Shutdown requested...")
        self._running = False
    
    def _cleanup(self):
        logger.info("Cleaning up...")
        self._consumer.close()
        self._redis.close()
        logger.info(f"Final stats: {self._metrics}")


if __name__ == "__main__":
    processor = StandaloneSignalProcessor()
    processor.run()
