"""
HIMARI L1 Signal Processor

Main entry point that consumes from Redpanda (Kafka), processes through
the signal layer components, and writes to Redis for Layer 2 consumption.

Integrates with:
- Your existing Redpanda topic: 'quality_scores' (output from Flink pipeline)
- Your existing Redis feature store

Usage:
    # Start processing
    python -m himari_l1.signal_processor
    
    # Or import and configure
    from himari_l1.signal_processor import SignalProcessor
    processor = SignalProcessor()
    processor.run()
"""

import json
import time
import logging
import signal
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading

import redis
from kafka import KafkaConsumer, KafkaProducer
import numpy as np

# Import our components - Tier 5 Primitives
from .primitives.welford import WelfordVariance
from .primitives.kalman import KalmanFilter, AdaptiveKalmanFilter
from .primitives.ultimate_smoother import UltimateSmoother
from .primitives.rls import RecursiveLeastSquares, RegressionChannel
from .primitives.garch import OnlineGARCH
from .primitives.hurst import MovingHurst
from .primitives.volume import SyntheticVolumeDelta, RelativeVolume, OrderBookImbalance
from .primitives.tdigest_quantiles import StreamingQuantiles
from .primitives.covariance import OnlineCovariance

# Tier 3 ML Layer
from .ml import LorentzianKNN, EnsembleFusion

# Tier 1 Signal Fusion
from .fusion import DempsterShafer

# Feature Vector Assembly
from .feature_vector import FeatureVectorAssembler

# Regime Detection
from .regime import StreamingHMM, RegimeState

from .config import (
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
    """
    Per-symbol processing state.
    
    Holds all streaming algorithm instances for a single trading pair.
    Complete HIMARI L1 5-tier architecture with 50D feature output.
    """
    symbol: str
    
    # === TIER 5: Streaming Primitives ===
    # Price tracking
    kalman: KalmanFilter = field(default_factory=lambda: KalmanFilter(0.01, 0.1))
    smoother: UltimateSmoother = field(default_factory=lambda: UltimateSmoother(20))
    
    # Volatility/Variance
    returns_welford: WelfordVariance = field(default_factory=WelfordVariance)
    garch: OnlineGARCH = field(default_factory=OnlineGARCH)
    
    # Volume microstructure
    volume_delta: SyntheticVolumeDelta = field(default_factory=SyntheticVolumeDelta)
    rvol: RelativeVolume = field(default_factory=RelativeVolume)
    obi: OrderBookImbalance = field(default_factory=OrderBookImbalance)
    
    # Distribution tracking
    tdigest: StreamingQuantiles = field(default_factory=StreamingQuantiles)
    
    # === TIER 4: DSP/Regime ===
    hurst: MovingHurst = field(default_factory=lambda: MovingHurst(100, 10))
    hmm: StreamingHMM = field(default_factory=lambda: StreamingHMM(3))
    regression: RegressionChannel = field(default_factory=lambda: RegressionChannel(0.99, 2.0))
    
    # === TIER 3: ML Prediction ===
    lorentzian: LorentzianKNN = field(default_factory=lambda: LorentzianKNN(k=20, feature_dim=15))
    ensemble: EnsembleFusion = field(default_factory=lambda: EnsembleFusion(
        num_models=4, model_names=['kalman', 'lorentzian', 'hmm', 'hurst']
    ))
    
    # === TIER 1: Signal Fusion ===
    dempster_shafer: DempsterShafer = field(default_factory=DempsterShafer)
    
    # === Tracking ===
    last_price: float = 0.0
    last_open: float = 0.0
    last_high: float = 0.0
    last_low: float = 0.0
    last_timestamp: int = 0
    tick_count: int = 0
    
    # Feature vector (50D output for Layer 2)
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(50))
    
    def compute_return(self, price: float) -> float:
        """Compute log return from last price."""
        if self.last_price <= 0:
            return 0.0
        return np.log(price / self.last_price)


class SignalProcessor:
    """
    Main L1 Signal Layer processor.
    
    Consumes market data from Kafka, processes through signal components,
    and writes signals to Redis for Layer 2 consumption.
    
    Architecture:
        Redpanda (quality_scores) → SignalProcessor → Redis (signals:*)
    """
    
    def __init__(self, config: Optional[L1Config] = None):
        """
        Initialize signal processor.
        
        Args:
            config: L1 configuration. Uses DEFAULT_CONFIG if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self._running = False
        self._states: Dict[str, SymbolState] = {}
        
        # Initialize Redis connection
        self._init_redis()
        
        # Initialize Kafka consumer
        self._init_kafka()
        
        # Metrics
        self._metrics = {
            'messages_processed': 0,
            'signals_published': 0,
            'errors': 0,
            'last_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        logger.info(f"SignalProcessor initialized for symbols: {self.config.symbols}")
    
    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        self._redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            ssl=REDIS_SSL,
            db=REDIS_DB,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        
        # Test connection
        try:
            self._redis.ping()
            logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def _init_kafka(self) -> None:
        """Initialize Kafka consumer."""
        self._consumer = KafkaConsumer(
            KAFKA_INPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_CONSUMER_GROUP,
            auto_offset_reset='latest',  # Start from latest on fresh start
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            max_poll_interval_ms=300000,
            session_timeout_ms=30000,
        )
        
        logger.info(f"Kafka consumer connected: {KAFKA_BOOTSTRAP}, topic: {KAFKA_INPUT_TOPIC}")
    
    def _get_state(self, symbol: str) -> SymbolState:
        """Get or create state for symbol, warm-starting from Redis if available."""
        if symbol not in self._states:
            state = SymbolState(symbol=symbol)
            
            # Try to restore from Redis
            try:
                state = self._restore_state(symbol, state)
            except Exception as e:
                logger.warning(f"Could not restore state for {symbol}: {e}")
            
            self._states[symbol] = state
            logger.info(f"Initialized state for {symbol}")
        
        return self._states[symbol]
    
    def _restore_state(self, symbol: str, state: SymbolState) -> SymbolState:
        """Restore algorithm state from Redis for warm restart."""
        # Try to restore Kalman
        kalman_key = RedisKeys.for_symbol(RedisKeys.STATE_KALMAN, symbol)
        kalman_json = self._redis.get(kalman_key)
        if kalman_json:
            state.kalman = KalmanFilter.from_json(kalman_json)
            logger.debug(f"Restored Kalman state for {symbol}")
        
        # Try to restore HMM
        hmm_key = RedisKeys.for_symbol(RedisKeys.STATE_HMM, symbol)
        hmm_json = self._redis.get(hmm_key)
        if hmm_json:
            state.hmm = StreamingHMM.from_json(hmm_json)
            logger.debug(f"Restored HMM state for {symbol}")
        
        # Try to restore Welford
        welford_key = RedisKeys.for_symbol(RedisKeys.STATE_WELFORD, symbol)
        welford_json = self._redis.get(welford_key)
        if welford_json:
            state.returns_welford = WelfordVariance.from_json(welford_json)
            logger.debug(f"Restored Welford state for {symbol}")
        
        # Try to restore GARCH
        garch_key = RedisKeys.for_symbol('himari:state:{symbol}:garch', symbol)
        garch_json = self._redis.get(garch_key)
        if garch_json:
            state.garch = OnlineGARCH.from_json(garch_json)
            logger.debug(f"Restored GARCH state for {symbol}")
        
        # Try to restore Lorentzian KNN
        knn_key = RedisKeys.for_symbol('himari:state:{symbol}:lorentzian', symbol)
        knn_json = self._redis.get(knn_key)
        if knn_json:
            state.lorentzian = LorentzianKNN.from_json(knn_json)
            logger.debug(f"Restored Lorentzian state for {symbol}")
        
        return state
    
    def _persist_state(self, symbol: str, state: SymbolState) -> None:
        """Persist algorithm state to Redis for warm restart."""
        pipe = self._redis.pipeline()
        
        # Kalman
        kalman_key = RedisKeys.for_symbol(RedisKeys.STATE_KALMAN, symbol)
        pipe.set(kalman_key, state.kalman.to_json())
        
        # HMM
        hmm_key = RedisKeys.for_symbol(RedisKeys.STATE_HMM, symbol)
        pipe.set(hmm_key, state.hmm.to_json())
        
        # Welford
        welford_key = RedisKeys.for_symbol(RedisKeys.STATE_WELFORD, symbol)
        pipe.set(welford_key, state.returns_welford.to_json())
        
        # GARCH
        garch_key = RedisKeys.for_symbol('himari:state:{symbol}:garch', symbol)
        pipe.set(garch_key, state.garch.to_json())
        
        # Lorentzian KNN
        knn_key = RedisKeys.for_symbol('himari:state:{symbol}:lorentzian', symbol)
        pipe.set(knn_key, state.lorentzian.to_json())
        
        pipe.execute()
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process market data through complete 5-tier signal pipeline.
        
        Tiers:
            5. Streaming Primitives (Welford, Kalman, GARCH, T-Digest)
            4. DSP/Regime (Hurst, HMM, Regression)
            3. ML Prediction (Lorentzian KNN, Ensemble)
            2. Volatility/Microstructure (Volume Delta, RVOL)
            1. Signal Fusion (Dempster-Shafer)
        
        Returns:
            Dict with signals + 50D feature vector for Layer 2
        """
        start_time = time.perf_counter()
        
        try:
            # === EXTRACT FIELDS ===
            symbol = message.get('symbol', '').upper()
            price = float(message.get('price', 0))
            timestamp = int(message.get('timestamp', 0))
            volume = float(message.get('volume', 0))
            quality_score = float(message.get('quality_score', 1.0))
            high = float(message.get('high', price))
            low = float(message.get('low', price))
            open_price = float(message.get('open', price))
            
            # Validate
            if not symbol or price <= 0:
                return None
            
            # Skip low quality data
            if quality_score < 0.5:
                logger.debug(f"Skipping low quality message for {symbol}: {quality_score}")
                return None
            
            # Get symbol state
            state = self._get_state(symbol)
            
            # Compute return
            ret = state.compute_return(price)
            
            # ============================================================
            # TIER 5: STREAMING PRIMITIVES
            # ============================================================
            
            # Price smoothing
            smoothed_price = state.smoother.update(price)
            kalman_price = state.kalman.update(price)
            
            # Variance tracking
            if abs(ret) < 0.5:  # Filter extreme moves
                state.returns_welford.update(ret)
            realized_vol = state.returns_welford.std * np.sqrt(252)
            
            # GARCH volatility
            garch_vol = state.garch.update(ret)
            garch_regime = state.garch.get_volatility_regime()
            
            # Price distribution
            state.tdigest.update(price)
            price_percentile = state.tdigest.relative_position(price)
            
            # ============================================================
            # TIER 4: DSP / REGIME DETECTION  
            # ============================================================
            
            # Hurst exponent (replaces Choppiness Index)
            hurst_value, hurst_regime = state.hurst.update(price)
            
            # HMM regime detection
            hmm_state, hmm_conf = state.hmm.update(ret)
            regime_probs = state.hmm.get_state_probabilities()
            regime_signal = state.hmm.get_regime_signal()
            
            # Trend regression
            reg_result = state.regression.update(state.tick_count, price)
            z_score = reg_result['z_score']
            mean_reversion_signal = 1 - 2 * reg_result['position']
            
            # ============================================================
            # TIER 2: VOLUME MICROSTRUCTURE
            # ============================================================
            
            # Volume delta (synthetic from OHLCV)
            vol_delta = state.volume_delta.update(open_price, high, low, price, volume)
            cvd = state.volume_delta.cumulative_delta
            
            # Relative volume (time-adjusted)
            rvol_zscore = state.rvol.update(volume, timestamp)
            
            # Order book imbalance (synthetic)
            obi = state.obi.update(open_price, high, low, price)
            
            # ============================================================
            # TIER 3: ML PREDICTION
            # ============================================================
            
            # Build mini feature vector for Lorentzian
            mini_features = np.array([
                ret,                              # Return
                (kalman_price - price) / price if price > 0 else 0,  # Kalman deviation
                hurst_value,                      # Hurst
                garch_vol / 0.05,                 # Normalized vol
                z_score / 3,                      # Normalized z-score
                regime_probs.get('BULL', 0.33),
                regime_probs.get('BEAR', 0.33),
                vol_delta / (volume + 1),         # Volume delta ratio
                rvol_zscore / 3,                  # RVOL normalized
                obi,                              # Order book imbalance
                price_percentile * 2 - 1,         # Price position
                0, 0, 0, 0                        # Padding
            ])
            
            # Lorentzian KNN prediction
            p_bullish, knn_confidence = state.lorentzian.predict(mini_features)
            
            # Ensemble fusion
            ensemble_predictions = [
                0.5 + (kalman_price - smoothed_price) / (smoothed_price * 0.02) if smoothed_price > 0 else 0.5,
                p_bullish,
                regime_probs.get('BULL', 0.33),
                state.hurst.get_momentum_weight() if hasattr(state.hurst, 'get_momentum_weight') else 0.5,
            ]
            ensemble_signal, ensemble_agreement = state.ensemble.predict(ensemble_predictions)
            
            # ============================================================
            # TIER 1: SIGNAL FUSION (Dempster-Shafer)
            # ============================================================
            
            state.dempster_shafer.reset()
            state.dempster_shafer.add_evidence({
                'bullish': p_bullish,
                'bearish': 1 - p_bullish,
            })
            state.dempster_shafer.add_evidence({
                'bullish': regime_probs.get('BULL', 0.33),
                'bearish': regime_probs.get('BEAR', 0.33),
                'neutral': regime_probs.get('RANGE', 0.34),
            })
            
            ds_decision, ds_belief, ds_uncertainty = state.dempster_shafer.get_decision()
            ds_confidence = state.dempster_shafer.get_confidence_weighted_signal()
            should_trade, trade_reason = state.dempster_shafer.should_trade()
            
            # ============================================================
            # ASSEMBLE 50D FEATURE VECTOR FOR LAYER 2
            # ============================================================
            
            feature_vector = np.zeros(50)
            
            # Trend features (0-7)
            feature_vector[0] = np.tanh((kalman_price - price) / price * 10) if price > 0 else 0
            feature_vector[1] = state.kalman.gain if hasattr(state.kalman, 'gain') else 0
            feature_vector[2] = np.tanh((smoothed_price - price) / price * 10) if price > 0 else 0
            feature_vector[3] = abs(reg_result['slope']) / 0.01
            feature_vector[4] = hurst_value * 2 - 1
            feature_vector[5] = 1.0 if hurst_regime == 'trending' else (-1.0 if hurst_regime == 'mean_reverting' else 0)
            feature_vector[6] = np.sign(price - smoothed_price)
            feature_vector[7] = hmm_conf
            
            # Momentum features (8-15)
            feature_vector[8] = p_bullish * 2 - 1
            feature_vector[9] = knn_confidence
            feature_vector[10] = ensemble_signal * 2 - 1
            feature_vector[11] = ensemble_agreement
            feature_vector[12] = 0  # Momentum divergence placeholder
            feature_vector[13] = np.tanh(ret * 50)
            feature_vector[14] = mean_reversion_signal
            feature_vector[15] = 0.5  # Momentum persistence
            
            # Volatility features (16-23)
            feature_vector[16] = min(garch_vol / 0.05, 1.0)
            feature_vector[17] = {'low': -1, 'normal': 0, 'high': 1}.get(garch_regime, 0)
            feature_vector[18] = regime_probs.get('BULL', 0.33)
            feature_vector[19] = regime_probs.get('BEAR', 0.33)
            feature_vector[20] = regime_probs.get('RANGE', 0.34)
            feature_vector[21] = realized_vol / (garch_vol + 0.001)
            feature_vector[22] = np.tanh(z_score / 2)
            feature_vector[23] = 0  # Vol trend
            
            # Volume features (24-33)
            feature_vector[24] = np.tanh(vol_delta / (volume + 1))
            feature_vector[25] = np.tanh(cvd / 1e6)
            feature_vector[26] = 0  # CVD divergence
            feature_vector[27] = np.tanh(rvol_zscore / 3)
            feature_vector[28] = obi
            feature_vector[29] = state.obi.imbalance_trend
            feature_vector[30] = 0  # Volume momentum
            feature_vector[31] = 1 if rvol_zscore > 2 else 0  # Volume breakout
            feature_vector[32] = max(0, obi)
            feature_vector[33] = max(0, -obi)
            
            # Statistical features (34-41)
            feature_vector[34] = price_percentile * 2 - 1
            feature_vector[35] = 0  # Realized vs GARCH
            feature_vector[36] = 0  # Skewness
            feature_vector[37] = 0  # Kurtosis
            feature_vector[38] = 0  # Correlation (need cross-asset)
            feature_vector[39] = 0  # Correlation strength
            feature_vector[40] = -feature_vector[5]  # Mean reversion = inverse trend
            feature_vector[41] = 0  # Autocorrelation
            
            # Meta features (42-45)
            feature_vector[42] = 1 - ds_uncertainty
            feature_vector[43] = state.dempster_shafer.conflict_level
            feature_vector[44] = 0  # Drawdown forecast
            feature_vector[45] = abs(regime_probs.get('BULL', 0.33) - regime_probs.get('BEAR', 0.33))
            
            # SMC/Microstructure (46-49)
            feature_vector[46] = 0.5  # Liquidity
            feature_vector[47] = 0    # Spread
            feature_vector[48] = 0    # Impact
            feature_vector[49] = 0    # Execution risk
            
            # Store feature vector in state
            state.feature_vector = feature_vector
            
            # ============================================================
            # BUILD OUTPUT SIGNALS
            # ============================================================
            
            # Momentum signal
            if smoothed_price > 0:
                momentum_signal = np.clip((price - smoothed_price) / smoothed_price * 100, -1, 1)
            else:
                momentum_signal = 0.0
            
            signals = {
                # === Core Signals ===
                'momentum': float(momentum_signal),
                'mean_reversion': float(mean_reversion_signal),
                'volatility': float(min(realized_vol, 2.0)),
                'garch_volatility': float(garch_vol),
                
                # === Regime ===
                'regime': ['BULL', 'BEAR', 'RANGE'][hmm_state],
                'regime_confidence': float(hmm_conf),
                'regime_signal': float(regime_signal),
                'hurst': float(hurst_value),
                'hurst_regime': hurst_regime,
                
                # === ML Predictions ===
                'p_bullish': float(p_bullish),
                'knn_confidence': float(knn_confidence),
                'ensemble_signal': float(ensemble_signal),
                'ensemble_agreement': float(ensemble_agreement),
                
                # === Fusion ===
                'ds_decision': ds_decision,
                'ds_confidence': float(ds_confidence),
                'ds_uncertainty': float(ds_uncertainty),
                'should_trade': should_trade,
                'trade_reason': trade_reason,
                
                # === Volume ===
                'volume_delta': float(vol_delta),
                'cvd': float(cvd),
                'rvol_zscore': float(rvol_zscore),
                'obi': float(obi),
                
                # === Derived ===
                'kalman_price': float(kalman_price),
                'smoothed_price': float(smoothed_price),
                'kalman_gain': float(state.kalman.gain),
                'trend_slope': float(reg_result['slope']),
                'z_score': float(z_score),
                'price_percentile': float(price_percentile),
                
                # === Recommendations ===
                'use_momentum': float(state.hmm.should_use_momentum()[1]),
                'use_mean_reversion': float(state.hmm.should_use_mean_reversion()[1]),
                
                # === Feature Vector (for L2) ===
                'feature_vector': feature_vector.tolist(),
                
                # === Metadata ===
                'timestamp': timestamp,
                'price': float(price),
                'symbol': symbol,
            }
            
            # Update state
            state.last_price = price
            state.last_open = open_price
            state.last_high = high
            state.last_low = low
            state.last_timestamp = timestamp
            state.tick_count += 1
            
            # Track latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._metrics['last_latency_ms'] = latency_ms
            
            return signals
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            self._metrics['errors'] += 1
            return None
    
    def publish_signals(self, signals: Dict[str, Any]) -> None:
        """
        Publish computed signals to Redis.
        
        Uses pipelining for efficiency.
        """
        symbol = signals['symbol']
        timestamp = signals['timestamp']
        
        pipe = self._redis.pipeline()
        
        # Latest signals hash
        latest_key = RedisKeys.for_symbol(RedisKeys.SIGNALS_LATEST, symbol)
        pipe.hset(latest_key, mapping={
            k: str(v) for k, v in signals.items()
        })
        
        # Individual signal keys (for efficient single-value reads)
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.SIGNAL_MOMENTUM, symbol),
            signals['momentum']
        )
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.SIGNAL_MEAN_REV, symbol),
            signals['mean_reversion']
        )
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.SIGNAL_VOLATILITY, symbol),
            signals['volatility']
        )
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.REGIME_STATE, symbol),
            signals['regime']
        )
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.REGIME_CONFIDENCE, symbol),
            signals['regime_confidence']
        )
        pipe.set(
            RedisKeys.for_symbol(RedisKeys.SIGNAL_TIMESTAMP, symbol),
            timestamp
        )
        
        pipe.execute()
        self._metrics['signals_published'] += 1
    
    def run(self) -> None:
        """
        Main processing loop.
        
        Consumes from Kafka, processes through signal layer, publishes to Redis.
        """
        self._running = True
        logger.info("Starting signal processor...")
        
        persist_interval = 60  # Persist state every 60 seconds
        last_persist_time = time.time()
        
        try:
            while self._running:
                # Poll for messages
                messages = self._consumer.poll(timeout_ms=100)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        # Process message
                        signals = self.process_message(record.value)
                        
                        if signals:
                            # Publish to Redis
                            self.publish_signals(signals)
                        
                        self._metrics['messages_processed'] += 1
                
                # Periodic state persistence
                if time.time() - last_persist_time > persist_interval:
                    self._persist_all_states()
                    last_persist_time = time.time()
                    self._log_metrics()
        
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()
    
    def _persist_all_states(self) -> None:
        """Persist all symbol states to Redis."""
        for symbol, state in self._states.items():
            try:
                self._persist_state(symbol, state)
            except Exception as e:
                logger.warning(f"Failed to persist state for {symbol}: {e}")
        
        logger.debug(f"Persisted state for {len(self._states)} symbols")
    
    def _log_metrics(self) -> None:
        """Log current metrics."""
        logger.info(
            f"Metrics: processed={self._metrics['messages_processed']}, "
            f"published={self._metrics['signals_published']}, "
            f"errors={self._metrics['errors']}, "
            f"latency={self._metrics['last_latency_ms']:.2f}ms"
        )
    
    def _shutdown_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
    
    def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up...")
        
        # Persist final state
        self._persist_all_states()
        
        # Close connections
        try:
            self._consumer.close()
        except:
            pass
        
        try:
            self._redis.close()
        except:
            pass
        
        logger.info("Shutdown complete")


def main():
    """Entry point for signal processor."""
    processor = SignalProcessor()
    processor.run()


if __name__ == '__main__':
    main()
