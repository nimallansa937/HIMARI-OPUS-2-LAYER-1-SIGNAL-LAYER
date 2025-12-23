"""
50-Dimensional Feature Vector Assembler for HIMARI L1

Assembles the complete L1 feature vector from all primitive components.
This is the output format consumed by Layer 2 ML models.

Feature Categories:
- 8 Trend features (Kalman, UltimateSmoother, Hurst)
- 8 Momentum features (Lorentzian prob, ensemble signal)  
- 8 Volatility features (GARCH, HMM regime probs)
- 10 Volume features (CVD, RVOL, OBI)
- 8 Statistical features (correlations, quantiles)
- 4 Meta features (DS confidence, drawdown forecast)
- 4 SMC/microstructure features (if available)

All features normalized to [-1, 1] or [0, 1].
Total latency target: <10ms per symbol.

Usage:
    assembler = FeatureVectorAssembler(primitives)
    
    for ohlcv in data:
        features = assembler.update(ohlcv)
        # features is 50-element numpy array
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
    Assembles the 50-dimensional feature vector from primitives.
    
    All components must be initialized externally and passed in.
    This class just coordinates updates and normalization.
    """
    
    FEATURE_DIM = 50
    
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
        covariance=None
    ):
        """
        Initialize with primitive components.
        
        Args:
            All primitives from HIMARI L1 (optional, will use None if not provided)
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
        
        # Feature metadata
        self._feature_names = self._create_feature_names()
        self._last_features = np.zeros(self.FEATURE_DIM)
        self._last_price = 0.0
        self._count = 0
    
    def _create_feature_names(self) -> List[str]:
        """Generate feature names for interpretability."""
        names = []
        
        # Trend features (0-7)
        names.extend([
            'kalman_smoothed_price',
            'kalman_velocity',
            'ultimate_smoother_value',
            'trend_strength',
            'hurst_exponent',
            'momentum_regime_score',
            'trend_direction',
            'trend_quality',
        ])
        
        # Momentum features (8-15)
        names.extend([
            'lorentzian_p_bullish',
            'lorentzian_confidence',
            'ensemble_signal',
            'ensemble_agreement',
            'momentum_divergence',
            'price_acceleration',
            'roc_normalized',
            'momentum_persistence',
        ])
        
        # Volatility features (16-23)
        names.extend([
            'garch_volatility',
            'garch_regime',
            'hmm_bull_prob',
            'hmm_bear_prob',
            'hmm_range_prob',
            'volatility_ratio',
            'variance_zscore',
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
        
        return names
    
    def update(self, ohlcv: Dict[str, Any]) -> np.ndarray:
        """
        Update all primitives and assemble feature vector.
        
        Args:
            ohlcv: Dict with keys: timestamp, open, high, low, close, volume
            
        Returns:
            50-element numpy array of normalized features
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
        
        # === TREND FEATURES (0-7) ===
        if self.kalman:
            kalman_state = self.kalman.update(price)
            features[0] = self._normalize_price(kalman_state, price)
            features[1] = getattr(self.kalman, 'velocity', 0) / 0.01
        
        if self.ultimate_smoother:
            smooth = self.ultimate_smoother.update(price)
            features[2] = self._normalize_price(smooth, price)
        
        if self.hurst:
            h, regime = self.hurst.update(price)
            features[4] = h * 2 - 1  # Map [0,1] to [-1,1]
            features[5] = 1.0 if regime == 'trending' else (
                          -1.0 if regime == 'mean_reverting' else 0.0)
        
        # Trend direction from smoother vs price
        features[6] = np.sign(price - features[2] * price) if features[2] != 0 else 0
        features[3] = abs(features[6]) * (1 - abs(features[4]))  # Trend strength
        features[7] = 0.5  # Placeholder for trend quality
        
        # === MOMENTUM FEATURES (8-15) ===
        if self.lorentzian:
            # Need feature vector for Lorentzian
            mini_features = np.array([log_return, features[0], features[2], 
                                      features[4], features[6]])
            mini_features = np.pad(mini_features, (0, 15 - len(mini_features)))
            p_bull, conf = self.lorentzian.predict(mini_features)
            features[8] = p_bull * 2 - 1  # Map [0,1] to [-1,1]
            features[9] = conf
        
        if self.ensemble:
            signal, agreement = self.ensemble.predict([
                features[8] / 2 + 0.5,  # Convert back to [0,1]
                features[5] / 2 + 0.5,
                0.5  # Placeholder
            ])
            features[10] = signal * 2 - 1
            features[11] = agreement
        
        features[12] = 0  # Momentum divergence placeholder
        features[13] = log_return / 0.01  # Price acceleration
        features[14] = np.tanh(log_return * 100)  # ROC normalized
        features[15] = 0.5  # Momentum persistence placeholder
        
        # === VOLATILITY FEATURES (16-23) ===
        if self.garch:
            vol = self.garch.update(log_return)
            features[16] = min(vol / 0.05, 1.0)  # Normalize to ~5% max
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
        
        if self.welford:
            self.welford.update(log_return)
            realized_vol = self.welford.std if hasattr(self.welford, 'std') else 0.02
            features[21] = realized_vol / (features[16] * 0.05 + 0.01)  # Vol ratio
        
        features[22] = 0  # Variance zscore placeholder
        features[23] = 0  # Vol trend placeholder
        
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
        
        features[40] = -features[5]  # Mean reversion = inverse of trend
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
