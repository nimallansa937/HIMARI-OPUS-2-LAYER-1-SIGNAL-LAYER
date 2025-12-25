"""
Integrated Signal Layer - Complete Layer 1 Signal Generation with SRM

Combines all 7 enhanced primitives:
1. StreamingHMM - Zero-lag regime detection
2. StreamingIndicators - O(1) technical indicators
3. WelfordOnlineStats - Memory-efficient statistics
4. MultiHorizonMomentum - Multi-timescale features
5. OrderBookImbalance - Academic-validated OBI
6. RegimeAwareSignalFusion - Regime-aware weighting
7. HybridSentiment - VADER + FinBERT (optional)

Plus SRM integration for systemic risk gating.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import logging

from .streaming_hmm import StreamingHMM, HMMConfig, MarketRegime
from .streaming_indicators import StreamingIndicators, IndicatorConfig
from .welford_stats import WelfordOnlineStats
from .multi_horizon_momentum import MultiHorizonMomentum, MomentumConfig
from .order_book_imbalance import OrderBookImbalance, OBIConfig
from .regime_fusion import RegimeAwareSignalFusion, FusionConfig

# Lazy import for sentiment (optional - requires torch/transformers)
try:
    from .hybrid_sentiment import HybridSentimentAnalyzer, HybridSentimentConfig
    _SENTIMENT_AVAILABLE = True
except (ImportError, OSError):
    _SENTIMENT_AVAILABLE = False
    HybridSentimentAnalyzer = None
    HybridSentimentConfig = None

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSignalOutput:
    """
    Final signal output with SRM risk gating.
    
    This is the complete output from Layer 1 that gets passed to Layer 2.
    """
    # Core signal
    composite_signal: float  # -1 to +1, final weighted signal
    
    # Regime information
    regime: str  # 'Bull', 'Bear', or 'Range'
    regime_confidence: float  # 0.0 to 1.0
    
    # SRM risk gating
    position_multiplier: float  # 0.0, 0.5, or 1.0 based on systemic risk
    srm_risk_score: float  # Current SRM composite risk score
    srm_action: str  # 'NORMAL', 'REDUCE', 'CLOSE_ONLY', or 'HALT'
    
    # Component signals (for monitoring/debugging)
    components: Dict[str, float]
    
    # Additional metadata
    update_count: int
    total_latency_ms: float


class IntegratedSignalLayer:
    """
    Complete Layer 1 signal generation with SRM integration.
    
    This is the master integration class that:
    1. Manages all 7 primitive components
    2. Orchestrates signal generation
    3. Applies regime-aware fusion
    4. Integrates SRM risk gating
    5. Returns unified signal output
    
    Example:
        layer = IntegratedSignalLayer(config, redis_client)
        
        signal = layer.update('BTCUSDT', ohlcv_data, orderbook_data)
        
        if signal.position_multiplier > 0:
            position_size = base_size * signal.composite_signal * signal.position_multiplier
    """
    
    def __init__(self, config, redis_client=None):
        """
        Initialize integrated signal layer.
        
        Args:
            config: EnhancedSignalConfig instance
            redis_client: Redis connection for SRM (optional)
        """
        self.config = config
        self.redis = redis_client
        self.update_count = 0
        
        logger.info("Initializing IntegratedSignalLayer...")
        
        # 1. Initialize HMM with config parameters
        hmm_config = HMMConfig(
            n_states=config.hmm_n_states,
            transition_persistence=config.hmm_transition_persistence,
            range_persistence=config.hmm_range_persistence,
            adaptive_enabled=config.hmm_adaptive_enabled,
            adaptive_lookback=config.hmm_adaptive_lookback,
            adaptive_frequency=config.hmm_adaptive_frequency,
            bull_mean=config.hmm_bull_mean,
            bull_std=config.hmm_bull_std,
            bear_mean=config.hmm_bear_mean,
            bear_std=config.hmm_bear_std,
            range_mean=config.hmm_range_mean,
            range_std=config.hmm_range_std
        )
        self.hmm = StreamingHMM(hmm_config) if config.hmm_enabled else None
        
        # 2. Initialize streaming indicators
        ind_config = IndicatorConfig(
            ema_periods=config.ema_periods,
            rsi_period=config.rsi_period,
            macd_fast=config.macd_fast,
            macd_slow=config.macd_slow,
            macd_signal=config.macd_signal,
            bb_period=config.bb_period,
            bb_std=config.bb_std,
            atr_period=config.atr_period
        )
        self.indicators = StreamingIndicators(ind_config) if config.indicators_enabled else None
        
        # 3. Initialize Welford statistics
        self.welford = WelfordOnlineStats(min_samples=config.welford_min_samples)
        
        # 4. Initialize multi-horizon momentum
        mom_config = MomentumConfig(
            horizons=config.momentum_horizons,
            normalization=config.momentum_normalization
        )
        self.momentum = MultiHorizonMomentum(mom_config) if config.momentum_enabled else None
        
        # 5. Initialize OBI
        obi_config = OBIConfig(
            levels=config.obi_levels,
            depth_percentage=config.obi_depth_percentage,
            ema_period=config.obi_ema_period
        )
        self.obi = OrderBookImbalance(obi_config) if config.obi_enabled else None
        
        # 6. Initialize regime-aware fusion
        fusion_config = FusionConfig(
            regime_weights=config.get_regime_weights(),
            confidence_threshold=config.fusion_confidence_threshold,
            min_regime_duration=config.fusion_min_regime_duration
        )
        self.fusion = RegimeAwareSignalFusion(
            self.hmm, 
            fusion_config
        ) if config.fusion_enabled and self.hmm else None
        
        # 7. Initialize hybrid sentiment (optional)
        self.sentiment = None
        if config.sentiment_enabled:
            if _SENTIMENT_AVAILABLE and HybridSentimentAnalyzer is not None:
                sent_config = HybridSentimentConfig(
                    vader_weight=config.sentiment_vader_weight,
                    transformer_weight=config.sentiment_finbert_weight,
                    transformer_model=config.sentiment_model,
                    max_batch_size=config.sentiment_batch_size
                )
                self.sentiment = HybridSentimentAnalyzer(sent_config)
            else:
                logger.warning("Sentiment requested but torch/transformers not available")
        
        # Register signals with fusion (after all components initialized)
        if self.fusion:
            self._register_signals()
        
        logger.info(f"✅ IntegratedSignalLayer initialized")
        logger.info(f"   HMM: {'✅' if self.hmm else '❌'}")
        logger.info(f"   Indicators: {'✅' if self.indicators else '❌'}")
        logger.info(f"   Momentum: {'✅' if self.momentum else '❌'}")
        logger.info(f"   OBI: {'✅' if self.obi else '❌'}")
        logger.info(f"   Fusion: {'✅' if self.fusion else '❌'}")
        logger.info(f"   Sentiment: {'✅' if self.sentiment else '❌'}")
    
    def _register_signals(self):
        """Register signal generators with fusion layer."""
        # Momentum signals
        self.fusion.register_signal('momentum_5', 'momentum', base_weight=1.0)
        self.fusion.register_signal('momentum_21', 'trend_following', base_weight=1.2)
        self.fusion.register_signal('momentum_63', 'trend_following', base_weight=1.5)

        # OBI signal
        self.fusion.register_signal('obi_normalized', 'volume', base_weight=1.5)

        # Mean reversion from RSI
        self.fusion.register_signal('rsi_signal', 'mean_reversion', base_weight=1.0)

        # ENHANCEMENT 1: Sentiment signals (register even if lag buffer not initialized yet)
        if self.sentiment:
            self.fusion.register_signal('sentiment_current', 'sentiment', base_weight=1.0)
            if self.sentiment.lag_buffer:
                self.fusion.register_signal('news_lag_30m', 'sentiment', base_weight=0.8)
                self.fusion.register_signal('news_lag_60m', 'sentiment', base_weight=0.7)
                self.fusion.register_signal('news_lag_90m', 'sentiment', base_weight=0.6)
                self.fusion.register_signal('news_lag_120m', 'sentiment', base_weight=0.5)
                self.fusion.register_signal('news_lag_180m', 'sentiment', base_weight=0.4)
                self.fusion.register_signal('sentiment_acceleration', 'sentiment', base_weight=0.9)
                logger.debug("Registered sentiment lag signals")
            else:
                logger.debug("Registered sentiment signal (no lag features)")

        logger.debug("Registered signals with fusion layer")
    
    def update(self, 
               symbol: str,
               ohlcv: Dict[str, float],
               orderbook: Optional[Dict] = None,
               sentiment_texts: Optional[list] = None) -> IntegratedSignalOutput:
        """
        Generate composite signal with SRM risk gating.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            ohlcv: Dict with keys: open, high, low, close, volume
            orderbook: Optional order book snapshot
            sentiment_texts: Optional list of sentiment texts to analyze
        
        Returns:
            IntegratedSignalOutput with all signal components and SRM gating
        """
        import time
        start_time = time.time()
        
        self.update_count += 1
        components = {}
        
        # 1. Update streaming indicators
        if self.indicators:
            ind_values = self.indicators.update(ohlcv)
            
            # RSI-based mean reversion signal
            if ind_values.get('rsi') is not None:
                rsi = ind_values['rsi']
                # Convert RSI to signal: oversold = bullish, overbought = bearish
                if rsi < 30:
                    components['rsi_signal'] = (30 - rsi) / 30  # 0 to 1
                elif rsi > 70:
                    components['rsi_signal'] = (70 - rsi) / 30  # -1 to 0
                else:
                    components['rsi_signal'] = 0.0
        
        # 2. Update multi-horizon momentum
        if self.momentum:
            mom_values = self.momentum.update(ohlcv['close'])
            
            # Add z-scored momentum features
            if mom_values.get('mom_5_z') is not None:
                components['momentum_5'] = mom_values['mom_5_z']
            if mom_values.get('mom_21_z') is not None:
                components['momentum_21'] = mom_values['mom_21_z']
            if mom_values.get('mom_63_z') is not None:
                components['momentum_63'] = mom_values['mom_63_z']
        
        # 3. Update OBI if order book available
        if self.obi and orderbook:
            obi_values = self.obi.update(orderbook)
            obi_signal = obi_values.get('obi_normalized')
            if obi_signal is not None:
                components['obi_normalized'] = obi_signal
        
        # 4. Update HMM with return
        price_return = None
        if self.hmm and self.update_count > 1:
            # Compute return (would need previous price stored)
            price_return = self.welford.get_mean() or 0.0  # Placeholder
            self.hmm.update(price_return)
        
        # 5. Update Welford stats with return
        if price_return is not None:
            self.welford.update(price_return)

        # 6. Get regime info for sentiment
        regime = 'Range'
        regime_conf = 0.33
        if self.hmm:
            regime = self.hmm.get_regime_label()
            regime_conf = float(self.hmm.state_probs.max())

        # 7. ENHANCEMENT: Process sentiment with regime context and lag features
        if self.sentiment and sentiment_texts:
            # Get ATR for volatility regime classification
            atr = ind_values.get('atr', 0.025) if self.indicators and ind_values else 0.025

            # Construct regime context for dynamic weighting
            regime_context = {
                'atr': atr,
                'social_zscore': 0.0,  # TODO: Add social volume tracking
                'market_regime': regime
            }

            # Analyze sentiment with regime context
            for text in sentiment_texts:
                sentiment_result = self.sentiment.analyze(
                    text,
                    regime_context=regime_context,
                    symbol=symbol,
                    source='news'
                )

                # Add current sentiment to components
                components['sentiment_current'] = sentiment_result['score']

                # ENHANCEMENT 1: Add lag features to components
                if 'lag_features' in sentiment_result:
                    for lag_name, lag_value in sentiment_result['lag_features'].items():
                        if lag_value != 0.0:  # Only add if buffer has sufficient data
                            components[lag_name] = lag_value

                # Log dynamic weights used (for monitoring)
                if 'weights_used' in sentiment_result:
                    logger.debug(f"Sentiment weights: {sentiment_result['weights_used']}")

        # 8. Fuse signals with regime-aware weighting
        composite_signal = 0.0

        if self.fusion and components:
            fusion_result = self.fusion.fuse(components, price_return)
            composite_signal = fusion_result['composite']
            regime = fusion_result['regime']
            regime_conf = fusion_result['confidence']
        elif components:
            # Fallback: simple average if fusion disabled
            composite_signal = sum(components.values()) / len(components)
        
        # 9. Get SRM risk score and apply gating
        srm_score, srm_action = self._get_srm_risk(symbol)
        position_mult = self._apply_srm_gating(composite_signal, srm_score)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return IntegratedSignalOutput(
            composite_signal=composite_signal,
            regime=regime,
            regime_confidence=regime_conf,
            position_multiplier=position_mult,
            srm_risk_score=srm_score,
            srm_action=srm_action,
            components=components,
            update_count=self.update_count,
            total_latency_ms=latency_ms
        )
    
    def _get_srm_risk(self, symbol: str) -> tuple:
        """
        Read SRM risk score from Redis.
        
        Returns:
            Tuple of (risk_score, action_string)
        """
        if not self.redis:
            return 0.0, 'NORMAL'
        
        try:
            key = f"srm:risk:{symbol}"
            score_str = self.redis.get(key)
            
            if score_str:
                score = float(score_str)
                
                # Determine action based on thresholds
                if score >= self.config.srm_halt_threshold:
                    action = 'HALT'
                elif score >= self.config.srm_close_only_threshold:
                    action = 'CLOSE_ONLY'
                elif score >= self.config.srm_reduce_threshold:
                    action = 'REDUCE'
                else:
                    action = 'NORMAL'
                
                return score, action
        
        except Exception as e:
            logger.error(f"Failed to get SRM risk for {symbol}: {e}")
        
        return 0.0, 'NORMAL'
    
    def _apply_srm_gating(self, signal: float, srm_score: float) -> float:
        """
        Apply SRM risk gating to position sizing.
        
        Args:
            signal: Composite signal value
            srm_score: SRM risk score
        
        Returns:
            Position multiplier: 0.0, 0.5, or 1.0
        """
        if srm_score >= self.config.srm_halt_threshold:
            logger.warning(f"SRM HALT triggered: score={srm_score:.2f}")
            return 0.0  # Emergency halt - no new positions
        
        elif srm_score >= self.config.srm_close_only_threshold:
            logger.warning(f"SRM CLOSE_ONLY triggered: score={srm_score:.2f}")
            return 0.0  # Close-only mode - no new positions
        
        elif srm_score >= self.config.srm_reduce_threshold:
            logger.info(f"SRM REDUCE triggered: score={srm_score:.2f}")
            return 0.5  # Reduce position sizing by 50%
        
        else:
            return 1.0  # Normal operation
    
    def get_stats(self) -> Dict:
        """Return statistics from all components."""
        stats = {
            'update_count': self.update_count,
            'hmm': self.hmm.get_stats() if self.hmm else None,
            'welford': self.welford.get_stats(),
        }
        
        if self.fusion:
            stats['fusion'] = {
                'current_regime': self.hmm.get_regime_label() if self.hmm else 'N/A',
                'confidence': float(self.hmm.state_probs.max()) if self.hmm else 0.0
            }
        
        return stats
