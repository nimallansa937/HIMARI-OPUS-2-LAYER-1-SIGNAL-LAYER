"""
Layer 1 Redis Consumer

Consumes signals from Signal Layer and publishes Layer 1 outputs via Redis.

Usage:
    from redis_consumer import Layer1RedisConsumer

    # Initialize
    consumer = Layer1RedisConsumer(symbols=["BTCUSDT"])

    # Option 1: Poll for latest signal
    signal = consumer.get_latest_signal("BTCUSDT")

    # Option 2: Subscribe to real-time signals
    def on_signal(data):
        # Process signal
        pass
    consumer.subscribe_to_signals(on_signal)
    consumer.start_listening()

    # Publish Layer 1 output
    consumer.publish_output("BTCUSDT", {
        "feature_vector": [...],
        "current_regime": "Bull",
        "directional_bias": 0.6,
    })
"""

import sys
import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
import numpy as np

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from redis_client import get_redis_client, HimariRedisClient
    from message_types import SignalMessage, Layer1Output, RegimeState
    from constants import (
        LAYER_1,
        CHANNEL_SIGNALS,
        CHANNEL_L1_DECISIONS,
        CHANNEL_REGIME,
        CHANNEL_L0_DATA_QUALITY,
        STATE_SIGNAL,
        STATE_REGIME,
        STATE_L1_OUTPUT,
        STATE_DATA_QUALITY,
        STREAM_SIGNALS,
        TTL_STATE,
        TTL_L1_OUTPUT,
        FEATURE_DIM_L1_CURRENT,
        SYMBOLS,
    )
except ImportError as e:
    print(f"Failed to import shared modules: {e}")
    print("Make sure the 'shared' directory exists in the HIMARI OPUS 2 folder")
    raise

logger = logging.getLogger(__name__)


class Layer1RedisConsumer:
    """
    Layer 1 Redis Consumer and Publisher.

    Responsibilities:
    - Consume signals from Signal Layer
    - Consume data quality from Layer 0
    - Consume regime state
    - Publish Layer 1 outputs (60D feature vectors)
    - Track regime transitions for edge detection
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize Layer 1 Redis consumer.

        Args:
            symbols: List of symbols to track (default: ["BTCUSDT"])
        """
        self.redis = get_redis_client(LAYER_1)
        self.symbols = symbols or SYMBOLS

        # Local cache for signals (to reduce Redis calls)
        self._signal_cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 1.0  # Cache for 1 second

        # Callbacks for real-time updates
        self._signal_callbacks: List[Callable] = []
        self._regime_callbacks: List[Callable] = []

        # Metrics
        self._signals_received = 0
        self._outputs_published = 0
        self._errors = 0

        # Running flag
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        logger.info(f"Layer1RedisConsumer initialized for symbols: {self.symbols}")

    # =========================================================================
    # CONSUME SIGNALS
    # =========================================================================

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest signal for symbol from Redis.

        Uses local cache to avoid excessive Redis calls.

        Args:
            symbol: Trading symbol

        Returns:
            Signal data dictionary or None
        """
        now = time.time()

        # Check cache
        if symbol in self._signal_cache:
            if now - self._cache_time.get(symbol, 0) < self._cache_ttl:
                return self._signal_cache[symbol]

        # Fetch from Redis
        state_key = STATE_SIGNAL.format(symbol=symbol)
        data = self.redis.get_state(state_key)

        if data:
            self._signal_cache[symbol] = data
            self._cache_time[symbol] = now
            return data

        logger.warning(f"No signal found for {symbol}")
        return None

    def get_regime_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current regime state for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Regime state dictionary or None
        """
        state_key = STATE_REGIME.format(symbol=symbol)
        return self.redis.get_state(state_key)

    def get_data_quality(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get data quality from Layer 0.

        Args:
            symbol: Trading symbol

        Returns:
            Data quality dictionary or None
        """
        state_key = STATE_DATA_QUALITY.format(symbol=symbol)
        return self.redis.get_state(state_key)

    def get_quality_score(self, symbol: str) -> float:
        """
        Get quality score from Layer 0.

        Args:
            symbol: Trading symbol

        Returns:
            float: Quality score (0.0-1.0), defaults to 0.70
        """
        quality = self.get_data_quality(symbol)
        if quality is None:
            return 0.70  # Default to minimum acceptable
        return quality.get("quality_score", 0.70)

    def should_use_data(self, symbol: str, min_threshold: float = 0.70) -> tuple:
        """
        Check if data quality is sufficient for processing.

        Args:
            symbol: Trading symbol
            min_threshold: Minimum acceptable quality score

        Returns:
            tuple: (use_data: bool, quality_score: float)
        """
        quality = self.get_data_quality(symbol)

        if quality is None:
            # No quality data - use conservative approach
            logger.warning(f"No Layer 0 quality data for {symbol}")
            return True, 0.70

        score = quality.get("quality_score", 0.0)
        grade = quality.get("quality_grade", "D")

        if score < min_threshold:
            logger.warning(
                f"Low quality data for {symbol}: "
                f"score={score:.4f}, grade={grade}"
            )
            return False, score

        return True, score

    def get_quality_weighted_value(self, symbol: str, value: float) -> float:
        """
        Weight a value by Layer 0 data quality.

        Args:
            symbol: Trading symbol
            value: Value to weight

        Returns:
            float: Quality-weighted value
        """
        quality = self.get_data_quality(symbol)

        if quality is None:
            return value * 0.85  # Conservative default

        score = quality.get("quality_score", 0.70)

        # Quality weighting: Grade A=1.0, B=0.95, C=0.85, D=0.50
        if score >= 0.95:
            return value * 1.0
        elif score >= 0.85:
            return value * 0.95
        elif score >= 0.70:
            return value * 0.85
        else:
            return value * 0.50

    def get_regime_features(self, symbol: str) -> np.ndarray:
        """
        Get regime features for feature vector.

        Returns: [prob_bull, prob_bear, prob_range, prob_crisis]
        """
        state = self.get_regime_state(symbol)

        if state is None:
            return np.array([0.33, 0.33, 0.34, 0.0])

        return np.array([
            state.get("prob_bull", 0.33),
            state.get("prob_bear", 0.33),
            state.get("prob_range", 0.34),
            state.get("prob_crisis", 0.0),
        ])

    def is_in_transition_window(self, symbol: str) -> bool:
        """Check if currently in a regime transition window."""
        state = self.get_regime_state(symbol)
        return state is not None and state.get("in_transition_window", False)

    def get_transition_type(self, symbol: str) -> Optional[str]:
        """Get current transition type if in window."""
        state = self.get_regime_state(symbol)
        if state and state.get("in_transition_window"):
            return state.get("transition_type")
        return None

    def is_profitable_transition(self, symbol: str) -> bool:
        """Check if current transition is profitable."""
        transition = self.get_transition_type(symbol)
        profitable_types = ["RANGE_TO_BULL", "NEUTRAL_TO_BULL", "RANGING_TO_TRENDING_UP"]
        return transition in profitable_types

    # =========================================================================
    # SUBSCRIBE TO UPDATES
    # =========================================================================

    def subscribe_to_signals(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to real-time signal updates.

        Args:
            callback: Function called with signal data when new signal arrives
        """
        self._signal_callbacks.append(callback)

        for symbol in self.symbols:
            channel = CHANNEL_SIGNALS.format(symbol=symbol)
            self.redis.subscribe(channel, self._handle_signal)

    def subscribe_to_regime(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to regime transition events.

        Args:
            callback: Function called when regime transition occurs
        """
        self._regime_callbacks.append(callback)

        for symbol in self.symbols:
            channel = CHANNEL_REGIME.format(symbol=symbol)
            self.redis.subscribe(channel, self._handle_regime)

    def _handle_signal(self, data: Dict[str, Any]) -> None:
        """Internal handler for signal messages."""
        self._signals_received += 1
        symbol = data.get("symbol", "UNKNOWN")

        # Update cache
        self._signal_cache[symbol] = data
        self._cache_time[symbol] = time.time()

        # Call registered callbacks
        for callback in self._signal_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
                self._errors += 1

    def _handle_regime(self, data: Dict[str, Any]) -> None:
        """Internal handler for regime transition events."""
        for callback in self._regime_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Regime callback error: {e}")
                self._errors += 1

    def start_listening(self) -> None:
        """Start listening for subscribed channels (blocking)."""
        self._running = True
        self.redis.start_listening()

    def start_listening_thread(self) -> threading.Thread:
        """Start listening in background thread."""
        self._running = True
        return self.redis.start_listening_thread()

    # =========================================================================
    # PUBLISH LAYER 1 OUTPUT
    # =========================================================================

    def publish_output(
        self,
        symbol: str,
        output_data: Dict[str, Any],
        feature_vector: Optional[List[float]] = None
    ) -> bool:
        """
        Publish Layer 1 output to Redis.

        Args:
            symbol: Trading symbol
            output_data: Output data dictionary containing:
                - current_regime: str
                - regime_confidence: float
                - is_regime_transition: bool
                - directional_bias: float (-1 to 1)
                - volatility_regime: str
                - And any other L1 outputs
            feature_vector: Optional 60D feature vector

        Returns:
            True if successful
        """
        try:
            timestamp = int(time.time() * 1000)

            # Get upstream data quality
            quality_data = self.get_data_quality(symbol)
            upstream_quality = quality_data.get("quality_score", 1.0) if quality_data else 1.0

            # Get regime state for transition info
            regime_state = self.get_regime_state(symbol)
            is_transition = False
            transition_type = None
            transition_hours = 0.0

            if regime_state:
                is_transition = regime_state.get("is_transition", False)
                transition_type = regime_state.get("transition_type")
                transition_hours = regime_state.get("window_hours_elapsed", 0.0)

            # Build message
            message = Layer1Output(
                timestamp=timestamp,
                source=LAYER_1,
                symbol=symbol,
                feature_vector=feature_vector or [],
                feature_names=output_data.get("feature_names", []),
                feature_staleness_flags=output_data.get("feature_staleness_flags", []),
                current_regime=output_data.get("current_regime", "RANGING"),
                regime_confidence=output_data.get("regime_confidence", 0.5),
                regime_duration_hours=output_data.get("regime_duration_hours", 0.0),
                is_regime_transition=is_transition,
                previous_regime=output_data.get("previous_regime"),
                transition_hours_ago=transition_hours,
                hifa_stage_passed=output_data.get("hifa_stage_passed", 0),
                hifa_deflated_sharpe=output_data.get("hifa_deflated_sharpe", 0.0),
                hifa_cpcv_folds_passed=output_data.get("hifa_cpcv_folds_passed", 0),
                hifa_monte_carlo_survival=output_data.get("hifa_monte_carlo_survival", 0.0),
                hifa_true_contribution=output_data.get("hifa_true_contribution", 0.0),
                hifa_neutralization_ic_ratio=output_data.get("hifa_neutralization_ic_ratio", 0.0),
                upstream_data_quality=upstream_quality,
                quality_weighted_confidence=output_data.get("regime_confidence", 0.5) * upstream_quality,
                epistemic_uncertainty=output_data.get("epistemic_uncertainty", 0.0),
                requires_human_escalation=output_data.get("requires_human_escalation", False),
                escalation_reason=output_data.get("escalation_reason"),
                causal_event_active=output_data.get("causal_event_active", False),
                causal_event_id=output_data.get("causal_event_id"),
                causal_context=output_data.get("causal_context"),
                active_strategy_id=output_data.get("active_strategy_id"),
                strategy_confidence=output_data.get("strategy_confidence", 0.0),
                strategy_sharpe_live=output_data.get("strategy_sharpe_live", 0.0),
                directional_bias=output_data.get("directional_bias", 0.0),
                volatility_regime=output_data.get("volatility_regime", "medium"),
                volatility_percentile=output_data.get("volatility_percentile", 50.0),
                processing_latency_ms=output_data.get("processing_latency_ms", 0.0),
                latency_budget_ms=50.0,
                latency_budget_exceeded=output_data.get("processing_latency_ms", 0.0) > 50.0,
            )

            message_dict = message.to_dict()

            # 1. Publish to channel
            self.redis.publish(CHANNEL_L1_DECISIONS, message_dict)

            # 2. Update state key
            state_key = STATE_L1_OUTPUT.format(symbol=symbol)
            self.redis.set_state(state_key, message_dict, ttl=TTL_L1_OUTPUT)

            # Update metrics
            self._outputs_published += 1

            logger.debug(f"Published L1 output for {symbol}: regime={message.current_regime}, bias={message.directional_bias:.3f}")
            return True

        except Exception as e:
            self._errors += 1
            logger.error(f"Failed to publish L1 output for {symbol}: {e}")
            return False

    # =========================================================================
    # HEARTBEAT
    # =========================================================================

    def start_heartbeat(self, interval: int = 5) -> None:
        """Start background heartbeat publishing."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.warning("Heartbeat already running")
            return

        self._running = True

        def heartbeat_loop():
            while self._running:
                try:
                    metrics = {
                        "signals_received": self._signals_received,
                        "outputs_published": self._outputs_published,
                        "errors": self._errors,
                        "symbols": self.symbols,
                    }
                    self.redis.publish_heartbeat(status="healthy", metrics=metrics)
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                time.sleep(interval)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            daemon=True,
            name="layer1-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("Started Layer 1 heartbeat")

    def stop_heartbeat(self) -> None:
        """Stop heartbeat publishing."""
        self._running = False

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "signals_received": self._signals_received,
            "outputs_published": self._outputs_published,
            "errors": self._errors,
            "redis_connected": self.redis.is_connected(),
        }

    def close(self) -> None:
        """Close consumer and cleanup."""
        self._running = False
        self.redis.stop_listening()
        logger.info("Layer1RedisConsumer closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_consumer(symbols: Optional[List[str]] = None) -> Layer1RedisConsumer:
    """Create and initialize Layer 1 Redis consumer."""
    consumer = Layer1RedisConsumer(symbols=symbols)
    consumer.start_heartbeat()
    return consumer


# Example integration
def example_integration():
    """
    Example showing how to integrate Redis with Layer 1.

    ```python
    from redis_consumer import Layer1RedisConsumer

    # Initialize
    redis_consumer = Layer1RedisConsumer(symbols=["BTCUSDT"])
    redis_consumer.start_heartbeat()

    # Option 1: Poll-based approach
    def process_cycle(symbol):
        # Get latest signal from Signal Layer
        signal = redis_consumer.get_latest_signal(symbol)
        if signal is None:
            return

        # Get regime state
        regime = redis_consumer.get_regime_state(symbol)

        # Check for profitable transition
        if redis_consumer.is_profitable_transition(symbol):
            print("In profitable transition window!")

        # Your Layer 1 processing here...
        feature_vector = compute_features(signal, regime)
        directional_bias = compute_bias(feature_vector)

        # Publish output
        redis_consumer.publish_output(symbol, {
            "current_regime": regime.get("current_regime", "RANGING"),
            "regime_confidence": regime.get("regime_confidence", 0.5),
            "directional_bias": directional_bias,
            "volatility_regime": "medium",
        }, feature_vector=feature_vector)

    # Option 2: Event-driven approach
    def on_new_signal(signal_data):
        symbol = signal_data.get("symbol")
        # Process immediately when new signal arrives
        process_cycle(symbol)

    redis_consumer.subscribe_to_signals(on_new_signal)
    redis_consumer.start_listening_thread()
    ```
    """
    pass


if __name__ == "__main__":
    # Test the consumer
    logging.basicConfig(level=logging.INFO)

    print("Testing Layer1RedisConsumer...")

    consumer = Layer1RedisConsumer(symbols=["BTCUSDT"])
    consumer.start_heartbeat()

    # Test getting signal
    signal = consumer.get_latest_signal("BTCUSDT")
    print(f"Latest signal: {signal}")

    # Test getting regime
    regime = consumer.get_regime_state("BTCUSDT")
    print(f"Regime state: {regime}")

    # Test publishing output
    test_output = {
        "current_regime": "Bull",
        "regime_confidence": 0.87,
        "directional_bias": 0.6,
        "volatility_regime": "medium",
    }
    test_features = [0.1] * 60  # 60D feature vector

    success = consumer.publish_output("BTCUSDT", test_output, feature_vector=test_features)
    print(f"Output published: {success}")

    print(f"Metrics: {consumer.get_metrics()}")

    # Cleanup
    consumer.close()
    print("Test complete!")
