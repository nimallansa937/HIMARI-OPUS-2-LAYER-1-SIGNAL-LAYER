"""
Layer 1 Consumer for Layer 0 Data Quality

Reads data quality messages from Redis and filters low-quality data.
Implements the Layer 0 -> Layer 1 integration as specified in INTEGRATION_GUIDE.md.

Usage:
    from layer0_consumer import Layer0Consumer, create_consumer

    # Initialize
    consumer = Layer0Consumer()

    # Check if data quality is sufficient
    use_data, score = consumer.should_use_data("BTCUSDT")

    # Weight values by quality
    weighted_confidence = consumer.get_quality_weighted_value("BTCUSDT", confidence)

    # Subscribe to real-time updates
    consumer.subscribe_to_quality_updates(callback)
"""

import sys
import os
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from redis_client import get_redis_client, HimariRedisClient
    from constants import (
        LAYER_1,
        CHANNEL_L0_DATA_QUALITY,
        STATE_DATA_QUALITY,
        SYMBOLS,
    )
    SHARED_AVAILABLE = True
except ImportError:
    SHARED_AVAILABLE = False
    # Fallback constants
    CHANNEL_L0_DATA_QUALITY = "himari:l0:data_quality"
    STATE_DATA_QUALITY = "state:{symbol}:data_quality"
    SYMBOLS = ["BTCUSDT"]

# Fallback Redis import
if not SHARED_AVAILABLE:
    try:
        import redis
    except ImportError:
        redis = None

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Quality score thresholds for data filtering."""
    grade_a: float = 0.95  # Excellent quality
    grade_b: float = 0.85  # Good quality
    grade_c: float = 0.70  # Acceptable quality (minimum)
    grade_d: float = 0.0   # Poor quality (reject)


@dataclass
class QualityWeights:
    """Quality-based weighting multipliers."""
    grade_a: float = 1.00  # Full weight for Grade A
    grade_b: float = 0.95  # 5% reduction for Grade B
    grade_c: float = 0.85  # 15% reduction for Grade C
    grade_d: float = 0.50  # 50% reduction for Grade D (or reject)
    no_data: float = 0.50  # Conservative default when no quality data


class Layer0Consumer:
    """
    Consumes Layer 0 data quality messages from Redis.

    Provides quality-aware data filtering and confidence weighting
    for Layer 1 processing.

    Features:
    - Poll-based quality retrieval
    - Real-time subscription support
    - Quality-weighted confidence adjustment
    - Configurable thresholds
    - Caching for performance
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        symbols: Optional[List[str]] = None,
        min_quality_threshold: float = 0.70,
        use_shared_client: bool = True
    ):
        """
        Initialize Layer 0 consumer.

        Args:
            redis_host: Redis host address
            redis_port: Redis port
            symbols: List of symbols to track
            min_quality_threshold: Minimum quality score to use data (Grade C)
            use_shared_client: Use shared HimariRedisClient if available
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.symbols = symbols or SYMBOLS
        self.min_quality_threshold = min_quality_threshold

        # Thresholds and weights
        self.thresholds = QualityThresholds()
        self.weights = QualityWeights()

        # Initialize Redis connection
        self._init_redis(use_shared_client)

        # Cache for quality data
        self._quality_cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 1.0  # Cache for 1 second

        # Subscription callbacks
        self._quality_callbacks: List[Callable] = []
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None

        # Metrics
        self._quality_checks = 0
        self._low_quality_rejections = 0
        self._cache_hits = 0

        logger.info(f"Layer0Consumer initialized for symbols: {self.symbols}")

    def _init_redis(self, use_shared_client: bool):
        """Initialize Redis connection."""
        if use_shared_client and SHARED_AVAILABLE:
            try:
                self.redis = get_redis_client(LAYER_1)
                self.using_shared = True
                logger.info("Using shared HimariRedisClient for Layer 0 consumer")
                return
            except Exception as e:
                logger.warning(f"Could not use shared client: {e}, falling back")

        # Fallback to basic Redis client
        self.using_shared = False
        if redis is not None:
            try:
                self.redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                )
                self.redis.ping()
                logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis = None
        else:
            logger.warning("Redis not available - Layer 0 consumer will use defaults")
            self.redis = None

    # =========================================================================
    # QUALITY RETRIEVAL
    # =========================================================================

    def get_latest_quality(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest data quality for symbol from Redis.

        Uses local cache to avoid excessive Redis calls.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            dict: Quality message or None if unavailable
        """
        if self.redis is None:
            return None

        now = time.time()

        # Check cache
        if symbol in self._quality_cache:
            if now - self._cache_time.get(symbol, 0) < self._cache_ttl:
                self._cache_hits += 1
                return self._quality_cache[symbol]

        # Fetch from Redis
        try:
            if self.using_shared:
                key = STATE_DATA_QUALITY.format(symbol=symbol)
                data = self.redis.get_state(key)
            else:
                key = f"state:{symbol}:data_quality"
                raw_data = self.redis.get(key)
                data = json.loads(raw_data) if raw_data else None

            if data:
                self._quality_cache[symbol] = data
                self._cache_time[symbol] = now
                return data

        except Exception as e:
            logger.error(f"Error fetching quality for {symbol}: {e}")

        logger.debug(f"No quality data found for {symbol}")
        return None

    def get_quality_score(self, symbol: str) -> float:
        """
        Get quality score for symbol.

        Args:
            symbol: Trading pair

        Returns:
            float: Quality score (0.0-1.0) or default 0.7
        """
        quality = self.get_latest_quality(symbol)
        if quality is None:
            return 0.70  # Default to minimum threshold
        return quality.get("quality_score", 0.70)

    def get_quality_grade(self, symbol: str) -> str:
        """
        Get quality grade for symbol.

        Args:
            symbol: Trading pair

        Returns:
            str: Grade ("A", "B", "C", or "D")
        """
        quality = self.get_latest_quality(symbol)
        if quality is None:
            return "C"  # Default to minimum acceptable
        return quality.get("quality_grade", "C")

    # =========================================================================
    # QUALITY FILTERING
    # =========================================================================

    def should_use_data(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if data quality is sufficient to use.

        This is the primary method for quality-based filtering.
        Call this before processing any market data.

        Args:
            symbol: Trading pair

        Returns:
            tuple: (use_data: bool, quality_score: float)
        """
        self._quality_checks += 1

        quality = self.get_latest_quality(symbol)

        if quality is None:
            # No quality data - use conservative approach
            logger.warning(f"No quality data for {symbol} - using conservative default")
            return True, 0.70  # Allow but with reduced confidence

        score = quality.get("quality_score", 0.0)
        grade = quality.get("quality_grade", "D")
        anomaly = quality.get("anomaly_detected", False)

        # Reject if below minimum threshold
        if score < self.min_quality_threshold:
            self._low_quality_rejections += 1
            logger.warning(
                f"Low quality data for {symbol}: "
                f"score={score:.4f}, grade={grade}"
            )
            return False, score

        # Warn if anomaly detected but still usable
        if anomaly:
            anomaly_type = quality.get("anomaly_type", "unknown")
            logger.warning(
                f"Anomaly detected for {symbol}: {anomaly_type} "
                f"(score={score:.4f})"
            )

        return True, score

    def is_data_fresh(self, symbol: str, max_age_ms: int = 5000) -> bool:
        """
        Check if quality data is fresh (not stale).

        Args:
            symbol: Trading pair
            max_age_ms: Maximum age in milliseconds

        Returns:
            bool: True if data is fresh
        """
        quality = self.get_latest_quality(symbol)
        if quality is None:
            return False

        timestamp = quality.get("timestamp", 0)
        current_time = int(time.time() * 1000)
        age_ms = current_time - timestamp

        return age_ms < max_age_ms

    # =========================================================================
    # QUALITY WEIGHTING
    # =========================================================================

    def get_quality_weighted_value(
        self,
        symbol: str,
        value: float
    ) -> float:
        """
        Weight a value by data quality.

        Use this to adjust confidence scores, signal strengths,
        or other values based on data quality.

        Args:
            symbol: Trading pair
            value: Original value to weight

        Returns:
            float: Quality-weighted value
        """
        quality = self.get_latest_quality(symbol)

        if quality is None:
            return value * self.weights.no_data

        score = quality.get("quality_score", 0.70)

        # Determine weight based on score
        if score >= self.thresholds.grade_a:
            weight = self.weights.grade_a
        elif score >= self.thresholds.grade_b:
            weight = self.weights.grade_b
        elif score >= self.thresholds.grade_c:
            weight = self.weights.grade_c
        else:
            weight = self.weights.grade_d

        return value * weight

    def get_quality_multiplier(self, symbol: str) -> float:
        """
        Get quality multiplier for position sizing.

        This is used by Layer 3 for quality-adjusted position sizing.

        Args:
            symbol: Trading pair

        Returns:
            float: Multiplier (0.5-1.0)
        """
        quality = self.get_latest_quality(symbol)

        if quality is None:
            return self.weights.no_data

        grade = quality.get("quality_grade", "C")

        multipliers = {
            "A": self.weights.grade_a,
            "B": self.weights.grade_b,
            "C": self.weights.grade_c,
            "D": self.weights.grade_d,
        }

        return multipliers.get(grade, self.weights.no_data)

    def adjust_confidence_threshold(
        self,
        symbol: str,
        base_threshold: float = 0.70
    ) -> float:
        """
        Adjust confidence threshold based on data quality.

        Higher thresholds for lower quality data means
        we require more certainty when data is unreliable.

        Args:
            symbol: Trading pair
            base_threshold: Base confidence threshold

        Returns:
            float: Adjusted threshold
        """
        quality = self.get_latest_quality(symbol)

        if quality is None:
            return base_threshold + 0.10  # More conservative

        score = quality.get("quality_score", 0.70)

        # Grade A: Lower threshold (more permissive)
        if score >= self.thresholds.grade_a:
            return base_threshold - 0.05
        # Grade B: Slightly higher
        elif score >= self.thresholds.grade_b:
            return base_threshold
        # Grade C: Higher threshold (more conservative)
        elif score >= self.thresholds.grade_c:
            return base_threshold + 0.05
        # Grade D: Much higher (very conservative)
        else:
            return min(base_threshold + 0.15, 0.95)

    # =========================================================================
    # REAL-TIME SUBSCRIPTION
    # =========================================================================

    def subscribe_to_quality_updates(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to real-time quality updates.

        Args:
            callback: Function called with quality messages
        """
        self._quality_callbacks.append(callback)

        if self.using_shared:
            self.redis.subscribe(CHANNEL_L0_DATA_QUALITY, self._handle_quality_message)
        elif self.redis is not None:
            # Start pubsub listener if not already running
            if self._listener_thread is None or not self._listener_thread.is_alive():
                self._start_pubsub_listener()

    def _handle_quality_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming quality message."""
        symbol = data.get("symbol", "UNKNOWN")

        # Update cache
        self._quality_cache[symbol] = data
        self._cache_time[symbol] = time.time()

        # Call registered callbacks
        for callback in self._quality_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Quality callback error: {e}")

    def _start_pubsub_listener(self) -> None:
        """Start pubsub listener thread for basic Redis client."""
        if self.redis is None:
            return

        self._running = True

        def listen():
            pubsub = self.redis.pubsub()
            pubsub.subscribe(CHANNEL_L0_DATA_QUALITY)

            for message in pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        self._handle_quality_message(data)
                    except Exception as e:
                        logger.error(f"Error parsing quality message: {e}")

            pubsub.unsubscribe()

        self._listener_thread = threading.Thread(
            target=listen,
            daemon=True,
            name="layer0-quality-listener"
        )
        self._listener_thread.start()
        logger.info("Started Layer 0 quality listener")

    def start_listening(self) -> None:
        """Start listening for quality updates (blocking)."""
        if self.using_shared:
            self.redis.start_listening()
        else:
            self._start_pubsub_listener()
            # Block main thread
            while self._running:
                time.sleep(1)

    def start_listening_thread(self) -> Optional[threading.Thread]:
        """Start listening in background thread."""
        if self.using_shared:
            return self.redis.start_listening_thread()
        else:
            self._start_pubsub_listener()
            return self._listener_thread

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_all_quality_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get quality states for all tracked symbols.

        Returns:
            dict: Symbol -> quality state mapping
        """
        states = {}
        for symbol in self.symbols:
            quality = self.get_latest_quality(symbol)
            if quality:
                states[symbol] = {
                    "quality_score": quality.get("quality_score", 0.0),
                    "quality_grade": quality.get("quality_grade", "D"),
                    "anomaly_detected": quality.get("anomaly_detected", False),
                    "primary_source_healthy": quality.get("primary_source_healthy", True),
                    "fallback_active": quality.get("fallback_active", False),
                }
        return states

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "quality_checks": self._quality_checks,
            "low_quality_rejections": self._low_quality_rejections,
            "cache_hits": self._cache_hits,
            "rejection_rate": (
                self._low_quality_rejections / max(self._quality_checks, 1)
            ),
            "symbols_tracked": len(self.symbols),
            "redis_connected": self.redis is not None,
            "using_shared_client": self.using_shared,
        }

    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        if self.redis is None:
            return False
        try:
            if self.using_shared:
                return self.redis.is_connected()
            else:
                self.redis.ping()
                return True
        except Exception:
            return False

    def close(self) -> None:
        """Close consumer and cleanup."""
        self._running = False
        logger.info("Layer0Consumer closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_consumer(
    symbols: Optional[List[str]] = None,
    min_quality_threshold: float = 0.70
) -> Layer0Consumer:
    """
    Create and initialize Layer 0 consumer.

    Args:
        symbols: List of symbols to track
        min_quality_threshold: Minimum acceptable quality score

    Returns:
        Layer0Consumer instance
    """
    return Layer0Consumer(
        symbols=symbols,
        min_quality_threshold=min_quality_threshold
    )


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

def example_layer1_integration():
    """
    Example showing how to integrate Layer 0 consumer with Layer 1.

    ```python
    from layer0_consumer import Layer0Consumer

    # Initialize at startup
    l0_consumer = Layer0Consumer(symbols=["BTCUSDT"])

    # In your main processing loop:
    def process_symbol(symbol: str, market_data: dict):
        # 1. Check data quality first
        use_data, quality_score = l0_consumer.should_use_data(symbol)

        if not use_data:
            logger.warning(f"Skipping {symbol} due to low quality: {quality_score:.4f}")
            return None

        # 2. Calculate your base confidence
        base_confidence = calculate_signal_confidence(market_data)

        # 3. Apply quality weighting
        weighted_confidence = l0_consumer.get_quality_weighted_value(
            symbol,
            base_confidence
        )

        # 4. Adjust thresholds if needed
        threshold = l0_consumer.adjust_confidence_threshold(symbol, 0.70)

        # 5. Make decision
        if weighted_confidence > threshold:
            execute_strategy()

        logger.info(
            f"{symbol}: Quality={quality_score:.4f}, "
            f"Confidence={base_confidence:.4f} → {weighted_confidence:.4f}"
        )

        return weighted_confidence
    ```
    """
    pass


if __name__ == "__main__":
    # Test the consumer
    logging.basicConfig(level=logging.INFO)

    print("Testing Layer0Consumer...")

    consumer = Layer0Consumer(symbols=["BTCUSDT"])

    # Test connection
    print(f"Connected: {consumer.is_connected()}")

    # Test quality check
    use_data, score = consumer.should_use_data("BTCUSDT")
    print(f"Should use data: {use_data}, Score: {score:.4f}")

    # Test quality weighting
    base_value = 0.75
    weighted = consumer.get_quality_weighted_value("BTCUSDT", base_value)
    print(f"Base value: {base_value:.4f}, Weighted: {weighted:.4f}")

    # Test threshold adjustment
    base_threshold = 0.70
    adjusted = consumer.adjust_confidence_threshold("BTCUSDT", base_threshold)
    print(f"Base threshold: {base_threshold:.4f}, Adjusted: {adjusted:.4f}")

    # Test multiplier
    multiplier = consumer.get_quality_multiplier("BTCUSDT")
    print(f"Quality multiplier: {multiplier:.4f}")

    # Test metrics
    print(f"Metrics: {consumer.get_metrics()}")

    # Cleanup
    consumer.close()
    print("Test complete!")
