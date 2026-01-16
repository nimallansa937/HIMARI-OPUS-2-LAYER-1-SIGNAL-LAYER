"""
60-Dimensional Feature Vector System

Provides type-safe feature storage and validation for trading signals.
Each feature has defined bounds, types, and update frequencies.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np


class FeatureType(Enum):
    """Dimensional typing prevents nonsense comparisons."""
    PRICE = "price"
    VOLUME = "volume"
    RATIO = "ratio"
    ZSCORE = "zscore"
    RATE = "rate"
    BOOLEAN = "boolean"
    COUNT = "count"


@dataclass
class FeatureSpec:
    """Specification for a single feature in the 60-dim vector."""
    index: int
    name: str
    type: FeatureType
    min_val: float
    max_val: float
    update_freq_ms: int
    description: str

    def validate(self, value: float) -> bool:
        """Check if value is within valid bounds."""
        if self.min_val == float('-inf') and self.max_val == float('inf'):
            return True
        return self.min_val <= value <= self.max_val

    def clamp(self, value: float) -> float:
        """Clamp value to valid bounds."""
        if self.min_val == float('-inf'):
            return min(value, self.max_val)
        if self.max_val == float('inf'):
            return max(value, self.min_val)
        return np.clip(value, self.min_val, self.max_val)


# Complete 60-Feature Vector Schema
FEATURE_SCHEMA: List[FeatureSpec] = [
    # ============== PRICE-DERIVED (0-14) ==============
    FeatureSpec(0, "close", FeatureType.PRICE, 0, float('inf'), 100, "Current close price"),
    FeatureSpec(1, "open", FeatureType.PRICE, 0, float('inf'), 100, "Current open price"),
    FeatureSpec(2, "high", FeatureType.PRICE, 0, float('inf'), 100, "Current high price"),
    FeatureSpec(3, "low", FeatureType.PRICE, 0, float('inf'), 100, "Current low price"),
    FeatureSpec(4, "sma_20", FeatureType.PRICE, 0, float('inf'), 1000, "20-period Simple Moving Average"),
    FeatureSpec(5, "sma_50", FeatureType.PRICE, 0, float('inf'), 1000, "50-period Simple Moving Average"),
    FeatureSpec(6, "ema_12", FeatureType.PRICE, 0, float('inf'), 1000, "12-period Exponential Moving Average"),
    FeatureSpec(7, "ema_26", FeatureType.PRICE, 0, float('inf'), 1000, "26-period Exponential Moving Average"),
    FeatureSpec(8, "bb_upper", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger Band upper"),
    FeatureSpec(9, "bb_lower", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger Band lower"),
    FeatureSpec(10, "bb_mid", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger Band middle"),
    FeatureSpec(11, "vwap", FeatureType.PRICE, 0, float('inf'), 1000, "Volume Weighted Average Price"),
    FeatureSpec(12, "atr_14", FeatureType.PRICE, 0, float('inf'), 1000, "14-period Average True Range"),
    FeatureSpec(13, "price_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Price z-score vs rolling mean"),
    FeatureSpec(14, "price_pct_change_1h", FeatureType.RATE, -0.5, 0.5, 1000, "1-hour price change percentage"),

    # ============== VOLUME-DERIVED (15-24) ==============
    FeatureSpec(15, "volume", FeatureType.VOLUME, 0, float('inf'), 100, "Current period volume"),
    FeatureSpec(16, "volume_sma_20", FeatureType.VOLUME, 0, float('inf'), 1000, "20-period volume SMA"),
    FeatureSpec(17, "volume_ratio", FeatureType.RATIO, 0, 10, 1000, "Volume / 20-period average"),
    FeatureSpec(18, "obv", FeatureType.VOLUME, float('-inf'), float('inf'), 1000, "On-Balance Volume"),
    FeatureSpec(19, "obv_slope", FeatureType.RATE, -1, 1, 1000, "OBV slope (normalized)"),
    FeatureSpec(20, "cvd", FeatureType.VOLUME, float('-inf'), float('inf'), 100, "Cumulative Volume Delta"),
    FeatureSpec(21, "cvd_slope", FeatureType.RATE, -1, 1, 1000, "CVD slope (normalized)"),
    FeatureSpec(22, "buy_volume_ratio", FeatureType.RATIO, 0, 1, 100, "Buy volume / total volume"),
    FeatureSpec(23, "large_trade_count", FeatureType.COUNT, 0, 1000, 1000, "Large trades in period"),
    FeatureSpec(24, "volume_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Volume z-score"),

    # ============== TECHNICAL INDICATORS (25-34) ==============
    FeatureSpec(25, "rsi_14", FeatureType.RATIO, 0, 100, 1000, "14-period RSI"),
    FeatureSpec(26, "rsi_30", FeatureType.RATIO, 0, 100, 1000, "30-period RSI"),
    FeatureSpec(27, "macd", FeatureType.ZSCORE, -10, 10, 1000, "MACD line"),
    FeatureSpec(28, "macd_signal", FeatureType.ZSCORE, -10, 10, 1000, "MACD signal line"),
    FeatureSpec(29, "macd_hist", FeatureType.ZSCORE, -5, 5, 1000, "MACD histogram"),
    FeatureSpec(30, "stoch_k", FeatureType.RATIO, 0, 100, 1000, "Stochastic %K"),
    FeatureSpec(31, "stoch_d", FeatureType.RATIO, 0, 100, 1000, "Stochastic %D"),
    FeatureSpec(32, "adx_14", FeatureType.RATIO, 0, 100, 1000, "14-period ADX (trend strength)"),
    FeatureSpec(33, "cci_20", FeatureType.ZSCORE, -300, 300, 1000, "20-period CCI"),
    FeatureSpec(34, "mfi_14", FeatureType.RATIO, 0, 100, 1000, "14-period Money Flow Index"),

    # ============== ORDER FLOW (35-44) ==============
    FeatureSpec(35, "bid_ask_spread", FeatureType.RATE, 0, 0.1, 100, "Spread as ratio of mid price"),
    FeatureSpec(36, "order_book_imbalance", FeatureType.RATIO, -1, 1, 100, "Order book bid/ask imbalance"),
    FeatureSpec(37, "bid_depth_5", FeatureType.VOLUME, 0, float('inf'), 100, "Bid depth at 5 levels"),
    FeatureSpec(38, "ask_depth_5", FeatureType.VOLUME, 0, float('inf'), 100, "Ask depth at 5 levels"),
    FeatureSpec(39, "depth_imbalance", FeatureType.RATIO, -1, 1, 100, "Depth imbalance at 5 levels"),
    FeatureSpec(40, "microprice", FeatureType.PRICE, 0, float('inf'), 100, "Volume-weighted mid price"),
    FeatureSpec(41, "trade_flow_imbalance", FeatureType.RATIO, -1, 1, 100, "Trade flow direction imbalance"),
    FeatureSpec(42, "large_order_pressure", FeatureType.RATIO, -1, 1, 1000, "Large order directional pressure"),
    FeatureSpec(43, "spread_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Spread z-score vs historical"),
    FeatureSpec(44, "liquidity_score", FeatureType.RATIO, 0, 1, 1000, "Composite liquidity score"),

    # ============== FUNDING & CARRY (45-49) ==============
    FeatureSpec(45, "funding_rate", FeatureType.RATE, -0.01, 0.01, 8*3600*1000, "Perpetual funding rate"),
    FeatureSpec(46, "funding_rate_zscore", FeatureType.ZSCORE, -3, 3, 8*3600*1000, "Funding rate z-score"),
    FeatureSpec(47, "open_interest", FeatureType.VOLUME, 0, float('inf'), 60000, "Open interest in contracts"),
    FeatureSpec(48, "oi_change_1h", FeatureType.RATE, -0.5, 0.5, 60000, "Open interest 1h change"),
    FeatureSpec(49, "long_short_ratio", FeatureType.RATIO, 0.1, 10, 60000, "Long/short account ratio"),

    # ============== SENTIMENT & CROSS-ASSET (50-54) ==============
    FeatureSpec(50, "fear_greed_index", FeatureType.RATIO, 0, 100, 24*3600*1000, "Fear & Greed Index"),
    FeatureSpec(51, "social_sentiment", FeatureType.RATIO, -1, 1, 3600*1000, "Social media sentiment"),
    FeatureSpec(52, "btc_dominance", FeatureType.RATIO, 0, 1, 3600*1000, "BTC market dominance"),
    FeatureSpec(53, "eth_btc_ratio", FeatureType.RATIO, 0, 1, 60000, "ETH/BTC price ratio"),
    FeatureSpec(54, "usdt_dominance", FeatureType.RATIO, 0, 0.2, 3600*1000, "USDT market cap share"),

    # ============== REGIME INDICATORS (55-59) ==============
    FeatureSpec(55, "regime_label", FeatureType.COUNT, 0, 4, 60000, "HMM regime label (0-4)"),
    FeatureSpec(56, "regime_confidence", FeatureType.RATIO, 0, 1, 60000, "Regime classification confidence"),
    FeatureSpec(57, "volatility_regime", FeatureType.COUNT, 0, 2, 60000, "Volatility regime (0=low,1=med,2=high)"),
    FeatureSpec(58, "trend_strength", FeatureType.RATIO, 0, 1, 60000, "Overall trend strength"),
    FeatureSpec(59, "regime_transition_prob", FeatureType.RATIO, 0, 1, 60000, "Probability of regime transition"),
]

# Build lookup dictionaries
FEATURE_BY_NAME: Dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SCHEMA}
FEATURE_BY_INDEX: Dict[int, FeatureSpec] = {spec.index: spec for spec in FEATURE_SCHEMA}

# Feature category ranges
FEATURE_CATEGORIES = {
    "price": (0, 14),
    "volume": (15, 24),
    "technical": (25, 34),
    "order_flow": (35, 44),
    "funding": (45, 49),
    "sentiment": (50, 54),
    "regime": (55, 59),
}


class FeatureVector:
    """
    60-dimensional feature vector with type safety and validation.

    Provides:
    - Type-checked feature access
    - Automatic clamping to valid ranges
    - Timestamp tracking for staleness detection
    - Normalization for ML models
    """

    def __init__(self):
        self.values = np.zeros(60, dtype=np.float32)
        self.timestamps = np.zeros(60, dtype=np.int64)
        self._staleness_threshold_ms = 60000  # 1 minute default

    def set(self, name: str, value: float, timestamp_ms: int) -> None:
        """Set feature value with validation and timestamp."""
        if name not in FEATURE_BY_NAME:
            raise KeyError(f"Unknown feature: {name}")

        spec = FEATURE_BY_NAME[name]
        clamped = spec.clamp(value)
        self.values[spec.index] = clamped
        self.timestamps[spec.index] = timestamp_ms

    def set_by_index(self, index: int, value: float, timestamp_ms: int) -> None:
        """Set feature by index."""
        if index not in FEATURE_BY_INDEX:
            raise IndexError(f"Invalid feature index: {index}")

        spec = FEATURE_BY_INDEX[index]
        clamped = spec.clamp(value)
        self.values[index] = clamped
        self.timestamps[index] = timestamp_ms

    def get(self, name: str) -> float:
        """Get feature value by name."""
        if name not in FEATURE_BY_NAME:
            raise KeyError(f"Unknown feature: {name}")
        return self.values[FEATURE_BY_NAME[name].index]

    def get_by_index(self, index: int) -> float:
        """Get feature value by index."""
        if 0 <= index < 60:
            return self.values[index]
        raise IndexError(f"Invalid feature index: {index}")

    def get_timestamp(self, name: str) -> int:
        """Get last update timestamp for feature."""
        if name not in FEATURE_BY_NAME:
            raise KeyError(f"Unknown feature: {name}")
        return self.timestamps[FEATURE_BY_NAME[name].index]

    def is_stale(self, name: str, current_time_ms: int) -> bool:
        """Check if feature is stale (not updated recently)."""
        spec = FEATURE_BY_NAME[name]
        age_ms = current_time_ms - self.timestamps[spec.index]
        return age_ms > max(spec.update_freq_ms * 10, self._staleness_threshold_ms)

    def get_stale_features(self, current_time_ms: int) -> List[str]:
        """Return list of stale feature names."""
        return [spec.name for spec in FEATURE_SCHEMA
                if self.is_stale(spec.name, current_time_ms)]

    def normalize(self) -> np.ndarray:
        """
        Return min-max normalized vector.
        Features with infinite bounds are left as-is.
        """
        normalized = np.zeros(60, dtype=np.float32)

        for spec in FEATURE_SCHEMA:
            val = self.values[spec.index]

            if spec.max_val == float('inf') or spec.min_val == float('-inf'):
                # Can't normalize unbounded features - use raw value
                normalized[spec.index] = val
            else:
                range_val = spec.max_val - spec.min_val
                if range_val > 0:
                    normalized[spec.index] = (val - spec.min_val) / range_val
                else:
                    normalized[spec.index] = 0.0

        return normalized

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary."""
        return {spec.name: float(self.values[spec.index])
                for spec in FEATURE_SCHEMA}

    def from_dict(self, data: Dict[str, float], timestamp_ms: int) -> None:
        """Import from dictionary."""
        for name, value in data.items():
            if name in FEATURE_BY_NAME:
                self.set(name, value, timestamp_ms)

    def get_category(self, category: str) -> np.ndarray:
        """Get features for a specific category."""
        if category not in FEATURE_CATEGORIES:
            raise KeyError(f"Unknown category: {category}")
        start, end = FEATURE_CATEGORIES[category]
        return self.values[start:end+1].copy()

    def copy(self) -> 'FeatureVector':
        """Create a deep copy."""
        new_fv = FeatureVector()
        new_fv.values = self.values.copy()
        new_fv.timestamps = self.timestamps.copy()
        return new_fv

    def __repr__(self) -> str:
        non_zero = np.count_nonzero(self.values)
        return f"FeatureVector(non_zero={non_zero}/60)"


def get_feature_indices_by_type(feature_type: FeatureType) -> List[int]:
    """Get all feature indices of a specific type."""
    return [spec.index for spec in FEATURE_SCHEMA if spec.type == feature_type]


def get_comparable_features(feature_name: str) -> List[str]:
    """Get list of features that can be compared to the given feature (same type)."""
    if feature_name not in FEATURE_BY_NAME:
        return []
    target_type = FEATURE_BY_NAME[feature_name].type
    return [spec.name for spec in FEATURE_SCHEMA
            if spec.type == target_type and spec.name != feature_name]
