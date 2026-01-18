"""
60-Dimensional Feature Vector Assembler for HIMARI L1

Assembles the complete L1 feature vector from all primitive components.
This is the output format consumed by Layer 2 ML models.

ZERO-LAG INDICATOR REPLACEMENTS:
| Current (Lagging)   | Best Zero-Lag Replacement           |
|---------------------|-------------------------------------|
| RSI 14              | Ehlers Fisher Transform             |
| MACD histogram      | Zero-Lag MACD (ZLEMA-based)         |
| Stochastic K/D      | Ehlers Stochastic                   |
| ADX 14              | Ehlers Instantaneous Trendline      |
| Bollinger %B        | Keltner Channel (ATR-based)         |
| ATR %               | Ehlers SuperSmoother ATR            |
| CCI 20              | DEMA-smoothed CCI                   |
| Williams %R         | Keep as-is (already minimal lag)    |
| ROC 10              | Ehlers Roofing Filter / Acceleration|
| EMA/SMA             | JMA, ALMA, HMA, or KAMA             |

Top 3 Priorities:
1. JMA (Jurik Moving Average) — best all-around smoothing
2. Ehlers Fisher Transform — RSI replacement with leading characteristics
3. HMA (Hull Moving Average) — fast, smooth, easy to implement

Feature Categories:
- 8 Trend features (JMA, HMA, KAMA, Ehlers Instantaneous Trendline, Hurst)
- 8 Momentum features (Fisher RSI, Ehlers Roofing, Ehlers Stochastic, DEMA CCI)
- 8 Volatility features (GARCH, HMM regime probs, Ehlers SuperSmoother ATR)
- 10 Volume features (CVD, RVOL, OBI)
- 8 Statistical features (correlations, quantiles)
- 4 Meta features (DS confidence, drawdown forecast)
- 4 SMC/microstructure features
- 10 Order Flow features (OBI real-time, CVD, VPIN, microprice)

All features normalized to [-1, 1] or [0, 1].
Total latency target: <12ms per symbol.

Usage:
    assembler = FeatureVectorAssembler(primitives)

    for ohlcv in data:
        features = assembler.update(ohlcv)
        # features is 60-element numpy array
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class OHLCVData:
    """Single OHLCV bar."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureVectorAssembler:
    """
    Assembles the 60-dimensional feature vector from primitives.
    
    All components must be initialized externally and passed in.
    This class just coordinates updates and normalization.
    """
    
    FEATURE_DIM = 60  # Expanded from 50 to include order flow
    
    def __init__(
        self,
        kalman=None,
        ultimate_smoother=None,
        garch=None,
        hmm=None,
        hurst=None,
        welford=None,
        volume_delta=None,
        rvol=None,
        obi=None,
        lorentzian=None,
        ensemble=None,
        dempster_shafer=None,
        tdigest=None,
        covariance=None,
        order_flow=None,
        streaming_indicators=None,  # NEW: StreamingIndicators with zero-lag indicators
    ):
        """
        Initialize with primitive components.

        Args:
            All primitives from HIMARI L1 (optional, will use None if not provided)
            streaming_indicators: StreamingIndicators instance with zero-lag indicators
                                  (JMA, HMA, KAMA, ALMA, Fisher RSI, MESA MACD,
                                   Ehlers Stochastic, Ehlers Trendline, DEMA CCI,
                                   Ehlers Roofing, Keltner Channels, Ehlers ATR)
        """
        self.kalman = kalman
        self.ultimate_smoother = ultimate_smoother
        self.garch = garch
        self.hmm = hmm
        self.hurst = hurst
        self.welford = welford
        self.volume_delta = volume_delta
        self.rvol = rvol
        self.obi = obi
        self.lorentzian = lorentzian
        self.ensemble = ensemble
        self.dempster_shafer = dempster_shafer
        self.tdigest = tdigest
        self.covariance = covariance
        self.order_flow = order_flow
        self.streaming_indicators = streaming_indicators  # NEW: Zero-lag indicators

        # Feature metadata
        self._feature_names = self._create_feature_names()
        self._last_features = np.zeros(self.FEATURE_DIM)
        self._last_price = 0.0
        self._count = 0
        self._last_indicator_values = {}  # Cache indicator values
    
    def _create_feature_names(self) -> List[str]:
        """Generate feature names for interpretability.

        Zero-lag indicator replacements:
        - RSI → Ehlers Fisher Transform
        - MACD → Zero-Lag MACD (MESA MACD)
        - Stochastic → Ehlers Stochastic
        - ADX → Ehlers Instantaneous Trendline
        - Bollinger %B → Keltner %B
        - ATR → Ehlers SuperSmoother ATR
        - CCI → DEMA-smoothed CCI
        - ROC → Ehlers Roofing Filter
        - EMA/SMA → JMA, HMA, KAMA, ALMA
        """
        names = []

        # Trend features (0-7) - Using JMA, HMA, KAMA, Ehlers Trendline
        names.extend([
            'jma_smoothed_price',      # JMA instead of Kalman (near-zero lag)
            'jma_velocity',            # JMA derivative
            'hma_value',               # Hull MA (fast, smooth)
            'ehlers_trend_strength',   # Ehlers Instantaneous Trendline (replaces ADX)
            'hurst_exponent',          # Keep - statistical measure
            'kama_adaptive_value',     # KAMA adaptive smoothing
            'ehlers_trend_direction',  # From Ehlers Trendline
            'alma_trend_quality',      # ALMA Gaussian-weighted
        ])

        # Momentum features (8-15) - Using Fisher RSI, Ehlers Roofing, Ehlers Stoch
        names.extend([
            'fisher_rsi',              # Ehlers Fisher Transform (replaces RSI)
            'fisher_rsi_signal',       # Fisher signal line
            'mesa_macd',               # Zero-Lag MACD (replaces MACD)
            'mesa_macd_histogram',     # MESA histogram
            'ehlers_stoch_k',          # Ehlers Stochastic (replaces Stoch K/D)
            'ehlers_roofing',          # Ehlers Roofing Filter (replaces ROC)
            'dema_cci',                # DEMA-smoothed CCI (replaces CCI)
            'williams_r',              # Keep as-is (minimal lag)
        ])

        # Volatility features (16-23) - Using Ehlers SuperSmoother ATR, Keltner
        names.extend([
            'garch_volatility',
            'garch_regime',
            'hmm_bull_prob',
            'hmm_bear_prob',
            'hmm_range_prob',
            'ehlers_atr_pct',          # Ehlers SuperSmoother ATR % (replaces ATR %)
            'keltner_pct_b',           # Keltner %B (replaces Bollinger %B)
            'volatility_trend',
        ])

        # Volume features (24-33)
        names.extend([
            'volume_delta',
            'cvd_normalized',
            'cvd_divergence',
            'rvol_zscore',
            'obi_imbalance',
            'obi_trend',
            'volume_momentum',
            'volume_breakout_score',
            'buying_pressure',
            'selling_pressure',
        ])

        # Statistical features (34-41)
        names.extend([
            'price_percentile',
            'realized_vol_vs_garch',
            'skewness_realized',
            'kurtosis_realized',
            'correlation_btc',
            'correlation_strength',
            'mean_reversion_score',
            'autocorrelation',
        ])

        # Meta features (42-45)
        names.extend([
            'dempster_shafer_confidence',
            'signal_conflict_level',
            'drawdown_forecast',
            'regime_stability',
        ])

        # SMC/Microstructure (46-49)
        names.extend([
            'liquidity_score',
            'spread_normalized',
            'impact_estimate',
            'execution_risk',
        ])

        # Order Flow features (50-59)
        names.extend([
            'orderflow_obi_current',
            'orderflow_obi_ema',
            'orderflow_cvd_normalized',
            'orderflow_cvd_divergence',
            'orderflow_microprice_dev',
            'orderflow_vpin',
            'orderflow_spread_zscore',
            'orderflow_lob_imbalance',
            'orderflow_trade_intensity',
            'orderflow_aggressive_ratio',
        ])

        return names
    
    def update(self, ohlcv: Dict[str, Any]) -> np.ndarray:
        """
        Update all primitives and assemble feature vector.

        Uses zero-lag indicators when streaming_indicators is provided:
        - JMA, HMA, KAMA, ALMA for trend smoothing
        - Fisher RSI for momentum oscillator
        - MESA MACD for momentum divergence
        - Ehlers Stochastic for overbought/oversold
        - Ehlers Instantaneous Trendline for trend strength (replaces ADX)
        - DEMA CCI for cyclical momentum
        - Ehlers Roofing Filter for ROC replacement
        - Keltner Channels for volatility bands (replaces Bollinger)
        - Ehlers SuperSmoother ATR for volatility

        Args:
            ohlcv: Dict with keys: timestamp, open, high, low, close, volume

        Returns:
            60-element numpy array of normalized features
        """
        self._count += 1

        price = ohlcv.get('close', ohlcv.get('price', 0))
        high = ohlcv.get('high', price)
        low = ohlcv.get('low', price)
        open_price = ohlcv.get('open', price)
        volume = ohlcv.get('volume', 0)
        timestamp = ohlcv.get('timestamp', 0)

        # Compute log return
        if self._last_price > 0:
            log_return = np.log(price / self._last_price)
        else:
            log_return = 0.0

        features = np.zeros(self.FEATURE_DIM)

        # Update streaming indicators if available (ZERO-LAG)
        ind_values = {}
        if self.streaming_indicators:
            ind_values = self.streaming_indicators.update({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
            self._last_indicator_values = ind_values

        # === TREND FEATURES (0-7) - Using JMA, HMA, KAMA, Ehlers Trendline ===

        # Feature 0: JMA smoothed price (replaces Kalman)
        if ind_values.get('jma_21') is not None:
            features[0] = self._normalize_price(ind_values['jma_21'], price)
        elif self.kalman:
            kalman_state = self.kalman.update(price)
            features[0] = self._normalize_price(kalman_state, price)

        # Feature 1: JMA velocity (price derivative)
        if ind_values.get('jma_5') is not None and ind_values.get('jma_10') is not None:
            features[1] = (ind_values['jma_5'] - ind_values['jma_10']) / (price * 0.01 + 1e-10)
        elif self.kalman:
            features[1] = getattr(self.kalman, 'velocity', 0) / 0.01

        # Feature 2: HMA value (Hull Moving Average - fast, smooth)
        if ind_values.get('hma_21') is not None:
            features[2] = self._normalize_price(ind_values['hma_21'], price)
        elif self.ultimate_smoother:
            smooth = self.ultimate_smoother.update(price)
            features[2] = self._normalize_price(smooth, price)

        # Feature 3: Ehlers Trend Strength (replaces ADX)
        if ind_values.get('ehlers_trend_strength') is not None:
            features[3] = min(1.0, ind_values['ehlers_trend_strength'] / 50)  # Normalize 0-50 to 0-1
        else:
            features[3] = 0.5

        # Feature 4: Hurst exponent
        if self.hurst:
            h, regime = self.hurst.update(price)
            features[4] = h * 2 - 1  # Map [0,1] to [-1,1]

        # Feature 5: KAMA adaptive value
        if ind_values.get('kama') is not None:
            features[5] = self._normalize_price(ind_values['kama'], price)
        else:
            features[5] = 0.0

        # Feature 6: Ehlers Trend Direction
        if ind_values.get('ehlers_trend_direction') is not None:
            features[6] = ind_values['ehlers_trend_direction']  # Already -1, 0, or 1
        else:
            features[6] = np.sign(price - features[2] * price) if features[2] != 0 else 0

        # Feature 7: ALMA trend quality
        if ind_values.get('alma') is not None:
            features[7] = self._normalize_price(ind_values['alma'], price)
        else:
            features[7] = 0.5

        # === MOMENTUM FEATURES (8-15) - Using Fisher RSI, MESA MACD, Ehlers Stoch ===

        # Feature 8: Fisher RSI (replaces standard RSI)
        if ind_values.get('fisher_rsi') is not None:
            features[8] = np.tanh(ind_values['fisher_rsi'])  # Already somewhat normalized
        elif self.lorentzian:
            mini_features = np.array([log_return, features[0], features[2],
                                      features[4], features[6]])
            mini_features = np.pad(mini_features, (0, 15 - len(mini_features)))
            p_bull, conf = self.lorentzian.predict(mini_features)
            features[8] = p_bull * 2 - 1

        # Feature 9: Fisher RSI Signal
        if ind_values.get('fisher_signal') is not None:
            features[9] = np.tanh(ind_values['fisher_signal'])
        else:
            features[9] = 0.0

        # Feature 10: MESA MACD (Zero-Lag MACD)
        if ind_values.get('mesa_macd') is not None:
            features[10] = np.tanh(ind_values['mesa_macd'] / (price * 0.01 + 1e-10))
        elif self.ensemble:
            signal, agreement = self.ensemble.predict([
                features[8] / 2 + 0.5,
                features[5] / 2 + 0.5,
                0.5
            ])
            features[10] = signal * 2 - 1

        # Feature 11: MESA MACD Histogram
        if ind_values.get('mesa_histogram') is not None:
            features[11] = np.tanh(ind_values['mesa_histogram'] / (price * 0.001 + 1e-10))
        else:
            features[11] = 0.0

        # Feature 12: Ehlers Stochastic K (replaces Stochastic K/D)
        if ind_values.get('ehlers_stoch_k') is not None:
            features[12] = (ind_values['ehlers_stoch_k'] - 50) / 50  # Map 0-100 to -1,1
        else:
            features[12] = 0.0

        # Feature 13: Ehlers Roofing Filter (replaces ROC)
        if ind_values.get('ehlers_roofing') is not None:
            features[13] = np.tanh(ind_values['ehlers_roofing'] * 100)
        else:
            features[13] = log_return / 0.01

        # Feature 14: DEMA CCI (replaces standard CCI)
        if ind_values.get('dema_cci') is not None:
            features[14] = np.tanh(ind_values['dema_cci'] / 200)  # CCI typically -200 to 200
        else:
            features[14] = np.tanh(log_return * 100)

        # Feature 15: Williams %R (kept as-is, minimal lag)
        if ind_values.get('williams_r') is not None:
            features[15] = (ind_values['williams_r'] + 50) / 50  # Map -100,0 to -1,1
        else:
            features[15] = 0.5

        # === VOLATILITY FEATURES (16-23) - Using Ehlers ATR, Keltner ===

        if self.garch:
            vol = self.garch.update(log_return)
            features[16] = min(vol / 0.05, 1.0)
            features[17] = {'low': -1, 'normal': 0, 'high': 1}.get(
                           self.garch.get_volatility_regime(), 0)

        if self.hmm:
            state_probs = self.hmm.update(log_return)
            if isinstance(state_probs, dict):
                features[18] = state_probs.get('BULL', 0.33)
                features[19] = state_probs.get('BEAR', 0.33)
                features[20] = state_probs.get('RANGE', 0.34)
            elif hasattr(state_probs, '__iter__'):
                probs = list(state_probs) if not isinstance(state_probs, (int, float)) else [0.33, 0.33, 0.34]
                features[18:21] = probs[:3] if len(probs) >= 3 else [0.33, 0.33, 0.34]

        # Feature 21: Ehlers SuperSmoother ATR % (replaces ATR %)
        if ind_values.get('atr_pct') is not None:
            features[21] = min(1.0, ind_values['atr_pct'] / 5)  # Normalize to ~5% max
        elif self.welford:
            self.welford.update(log_return)
            realized_vol = self.welford.std if hasattr(self.welford, 'std') else 0.02
            features[21] = realized_vol / (features[16] * 0.05 + 0.01)

        # Feature 22: Keltner %B (replaces Bollinger %B)
        if ind_values.get('keltner_pct_b') is not None:
            features[22] = ind_values['keltner_pct_b'] * 2 - 1  # Map 0-1 to -1,1
        else:
            features[22] = 0.0

        # Feature 23: Volatility trend
        features[23] = 0

        # === VOLUME FEATURES (24-33) ===
        if self.volume_delta:
            delta = self.volume_delta.update(open_price, high, low, price, volume)
            features[24] = np.tanh(delta / (volume + 1))
            features[25] = np.tanh(self.volume_delta.cumulative_delta / 1e6)

        if self.rvol:
            zscore = self.rvol.update(volume, timestamp)
            features[27] = np.tanh(zscore / 3)

        if self.obi:
            imb = self.obi.update(open_price, high, low, price)
            features[28] = imb
            features[29] = self.obi.imbalance_trend

        features[26] = 0  # CVD divergence
        features[30] = 0  # Volume momentum
        features[31] = 1 if features[27] > 0.5 else 0  # Volume breakout
        features[32] = max(0, features[28])  # Buying pressure
        features[33] = max(0, -features[28])  # Selling pressure

        # === STATISTICAL FEATURES (34-41) ===
        if self.tdigest:
            self.tdigest.update(price)
            features[34] = self.tdigest.relative_position(price) * 2 - 1

        features[35] = 0  # Realized vs GARCH
        features[36] = 0  # Skewness
        features[37] = 0  # Kurtosis

        if self.covariance:
            features[38] = self.covariance.correlation
            features[39] = abs(self.covariance.correlation)

        features[40] = -features[6]  # Mean reversion = inverse of trend direction
        features[41] = 0  # Autocorrelation

        # === META FEATURES (42-45) ===
        if self.dempster_shafer:
            self.dempster_shafer.reset()
            self.dempster_shafer.add_evidence({
                'bullish': features[8] / 2 + 0.5,
                'bearish': (1 - features[8]) / 2,
            })
            _, belief, uncertainty = self.dempster_shafer.get_decision()
            features[42] = 1 - uncertainty
            features[43] = self.dempster_shafer.conflict_level

        features[44] = 0  # Drawdown forecast
        features[45] = abs(features[18] - features[19])  # Regime stability

        # === SMC/MICROSTRUCTURE (46-49) ===
        features[46] = 0.5  # Liquidity score placeholder
        features[47] = 0  # Spread normalized
        features[48] = 0  # Impact estimate
        features[49] = 0  # Execution risk

        # === ORDER FLOW FEATURES (50-59) ===
        if self.order_flow:
            order_flow_vec = self.order_flow.get_feature_vector()
            features[50:60] = order_flow_vec
        else:
            features[50] = 0  # obi_current
            features[51] = 0  # obi_ema
            features[52] = 0  # cvd_normalized
            features[53] = 0  # cvd_divergence
            features[54] = 0  # microprice_dev
            features[55] = 0  # vpin
            features[56] = 0  # spread_zscore
            features[57] = 0  # lob_imbalance
            features[58] = 0  # trade_intensity
            features[59] = 0.5  # aggressive_ratio

        # Store and return
        self._last_features = features
        self._last_price = price

        return features
    
    def _normalize_price(self, value: float, reference: float) -> float:
        """Normalize price-based value to [-1, 1] relative to reference."""
        if reference == 0:
            return 0
        return np.tanh((value - reference) / reference * 10)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        return self._feature_names.copy()
    
    def get_feature_dict(self) -> Dict[str, float]:
        """Return features as named dictionary."""
        return dict(zip(self._feature_names, self._last_features))
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of current features."""
        return {
            'n_nonzero': np.sum(self._last_features != 0),
            'mean': np.mean(self._last_features),
            'std': np.std(self._last_features),
            'min': np.min(self._last_features),
            'max': np.max(self._last_features),
        }
    
    def validate_features(self) -> Dict[str, Any]:
        """Validate features are in expected ranges."""
        f = self._last_features
        
        issues = []
        
        # Check for NaN/Inf
        if np.any(np.isnan(f)):
            issues.append('contains_nan')
        if np.any(np.isinf(f)):
            issues.append('contains_inf')
        
        # Check ranges
        if np.any(f > 10) or np.any(f < -10):
            issues.append('extreme_values')
        
        # Check too many zeros
        if np.sum(f == 0) > 30:
            issues.append('many_zeros')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
        }
    
    def __repr__(self) -> str:
        return f"FeatureVectorAssembler(dim={self.FEATURE_DIM}, count={self._count})"
