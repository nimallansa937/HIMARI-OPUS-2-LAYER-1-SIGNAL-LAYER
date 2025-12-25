"""
Multi-Asset Sentiment Manager - Scalable Model Management

Manages sentiment models across multiple assets with model pooling
and transfer learning support.

Enhanced with Multi-Model Ensemble (Phase 2):
- CryptoBERT for social media
- ModernFinBERT for news
- FinTwitBERT for ensemble voting
- Automatic source-based routing

Enhancement 7 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
"""

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for multi-model availability
try:
    from .multi_model_sentiment import (
        MultiModelSentimentAnalyzer,
        EnsembleConfig,
        EnsembleResult,
        ModelType,
        create_phase1_analyzer,
        create_phase2_analyzer,
        create_phase3_analyzer,
    )
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    logger.warning("Multi-model sentiment not available, falling back to hybrid analyzer")


class ModelVariant(Enum):
    """Model variant types for transfer learning."""
    BTC_BASE = "btc_base"                 # Use BTC model directly
    LIGHT_FINETUNED = "light_finetuned"   # BTC model + light fine-tuning
    DEDICATED = "dedicated"                # Separate dedicated model
    MULTI_MODEL = "multi_model"            # NEW: Multi-model ensemble (Phase 2)


@dataclass
class AssetConfig:
    """Configuration for a specific asset."""
    symbol: str
    model_variant: ModelVariant = ModelVariant.MULTI_MODEL  # DEFAULT to multi-model
    sentiment_sources: List[str] = field(default_factory=lambda: ['news', 'twitter'])
    model_path: Optional[str] = None
    
    # Asset-specific lag horizons (optional override)
    lag_horizons: Optional[Dict[str, List[int]]] = None
    
    # Expected correlation with BTC (for monitoring)
    expected_btc_correlation: float = 0.9
    
    # Use ensemble voting (Phase 2)
    use_ensemble: bool = True


# Pre-configured asset configs based on research
# Updated to use MULTI_MODEL by default
DEFAULT_ASSET_CONFIGS = {
    'BTCUSDT': AssetConfig(
        symbol='BTCUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter', 'reddit'],
        expected_btc_correlation=1.0,
        use_ensemble=True
    ),
    'ETHUSDT': AssetConfig(
        symbol='ETHUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.94,
        use_ensemble=True
    ),
    'SOLUSDT': AssetConfig(
        symbol='SOLUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.82,
        use_ensemble=True
    ),
    'BNBUSDT': AssetConfig(
        symbol='BNBUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.88,
        use_ensemble=True
    ),
    'ADAUSDT': AssetConfig(
        symbol='ADAUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.78,
        use_ensemble=True
    ),
    'DOGEUSDT': AssetConfig(
        symbol='DOGEUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter', 'reddit'],
        expected_btc_correlation=0.70,
        use_ensemble=True
    ),
    'XRPUSDT': AssetConfig(
        symbol='XRPUSDT',
        model_variant=ModelVariant.MULTI_MODEL,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.75,
        use_ensemble=True
    ),
}


class MultiAssetSentimentManager:
    """
    Manage sentiment models across multiple assets efficiently.
    
    Features:
    - Multi-model ensemble (Phase 2): CryptoBERT + ModernFinBERT + FinTwitBERT
    - Source-based routing (social → CryptoBERT, news → ModernFinBERT)
    - Ensemble voting with confidence levels
    - Model pooling (share models across assets)
    - Memory budget enforcement
    
    Example:
        manager = MultiAssetSentimentManager()
        
        # Analyze with auto-routing
        result = manager.analyze('BTCUSDT', 'BTC breaking $50k!', source='twitter')
        # Returns: {'score': 0.814, 'label': 'bullish', 'confidence': 0.81, ...}
        
        # Analyze with full ensemble voting
        ensemble = manager.analyze_ensemble('Breaking: SEC approves ETF')
        # Returns: {'score': 0.95, 'agreement': 1.0, 'recommendation': 'FULL', ...}
    """
    
    def __init__(
        self,
        asset_configs: Optional[Dict[str, AssetConfig]] = None,
        max_concurrent_models: int = 3,
        btc_model_path: str = "./models/finbert-crypto-finetuned",
        phase: int = 2,  # NEW: Default to Phase 2
    ):
        """
        Initialize multi-asset sentiment manager.
        
        Args:
            asset_configs: Per-asset configurations
            max_concurrent_models: Maximum model instances in memory
            btc_model_path: Path to BTC fine-tuned model
            phase: Multi-model phase (1, 2, or 3)
        """
        self.asset_configs = asset_configs or DEFAULT_ASSET_CONFIGS
        self.max_concurrent_models = max_concurrent_models
        self.btc_model_path = btc_model_path
        self.phase = phase
        
        # Model instances
        self._multi_model_analyzer = None  # NEW: Shared multi-model analyzer
        self._btc_base_analyzer = None     # Fallback: BTC base model
        self._light_models: Dict[str, Any] = {}
        self._dedicated_models: Dict[str, Any] = {}
        
        # Usage tracking for LRU eviction
        self._model_usage: Dict[str, int] = {}
        self._total_usage: int = 0
        
        # Initialize multi-model analyzer (shared across all assets)
        self._init_multi_model_analyzer()
        
        logger.info(f"MultiAssetSentimentManager initialized with {len(self.asset_configs)} assets")
        logger.info(f"Multi-model Phase: {self.phase}")
    
    def _init_multi_model_analyzer(self) -> None:
        """Initialize the multi-model ensemble analyzer."""
        if not MULTI_MODEL_AVAILABLE:
            logger.warning("Multi-model not available, will use fallback")
            return
        
        try:
            if self.phase == 1:
                self._multi_model_analyzer = create_phase1_analyzer()
            elif self.phase == 2:
                self._multi_model_analyzer = create_phase2_analyzer()
            elif self.phase == 3:
                self._multi_model_analyzer = create_phase3_analyzer()
            else:
                self._multi_model_analyzer = create_phase2_analyzer()
            
            logger.info(f"✓ Multi-model analyzer initialized (Phase {self.phase})")
        except Exception as e:
            logger.error(f"Failed to initialize multi-model analyzer: {e}")
            self._multi_model_analyzer = None
    
    def get_analyzer(self, symbol: str):
        """
        Get sentiment analyzer for an asset.
        
        Returns multi-model analyzer for MULTI_MODEL variant,
        falls back to BTC base for others.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Analyzer instance (multi-model or fallback)
        """
        config = self.asset_configs.get(symbol)
        
        if not config:
            logger.warning(f"No config for {symbol}, using multi-model analyzer")
            return self._multi_model_analyzer or self._get_btc_base_analyzer()
        
        # Update usage
        self._total_usage += 1
        self._model_usage[symbol] = self._total_usage
        
        if config.model_variant == ModelVariant.MULTI_MODEL:
            if self._multi_model_analyzer:
                return self._multi_model_analyzer
            else:
                return self._get_btc_base_analyzer()
        
        elif config.model_variant == ModelVariant.BTC_BASE:
            return self._get_btc_base_analyzer()
        
        elif config.model_variant == ModelVariant.LIGHT_FINETUNED:
            return self._get_light_finetuned(symbol, config)
        
        else:  # DEDICATED
            return self._get_dedicated(symbol, config)
    
    def analyze(
        self, 
        symbol: str, 
        text: str, 
        source: str = "unknown",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for an asset with source-based routing.
        
        Args:
            symbol: Trading symbol
            text: Text to analyze
            source: Source type (twitter, reddit, bloomberg, etc.)
            **kwargs: Additional args
            
        Returns:
            Dict with score, label, confidence, and model used
        """
        config = self.asset_configs.get(symbol)
        
        # Use multi-model if available and configured
        if (config and config.model_variant == ModelVariant.MULTI_MODEL 
            and self._multi_model_analyzer):
            
            result = self._multi_model_analyzer.analyze(text, source)
            
            if result:
                return {
                    'score': result.score,
                    'label': result.label,
                    'confidence': result.confidence,
                    'model': result.model_name,
                    'latency_ms': result.latency_ms,
                    'symbol': symbol,
                    'source': source,
                    'multi_model': True,
                }
        
        # Fallback to legacy analyzer
        analyzer = self.get_analyzer(symbol)
        
        if analyzer is None:
            return {
                'score': 0.0, 
                'label': 'neutral', 
                'confidence': 0.0,
                'error': 'no_analyzer',
                'symbol': symbol,
            }
        
        return analyzer.analyze(text, symbol=symbol, **kwargs)
    
    def analyze_ensemble(
        self, 
        text: str, 
        symbol: str = "BTCUSDT"
    ) -> Dict[str, Any]:
        """
        Analyze with full ensemble voting (Phase 2+).
        
        Returns detailed result with agreement rate and position recommendation.
        
        Args:
            text: Text to analyze
            symbol: Trading symbol for context
            
        Returns:
            Dict with ensemble voting results
        """
        if not self._multi_model_analyzer:
            return self.analyze(symbol, text)
        
        try:
            result = self._multi_model_analyzer.analyze_ensemble(text)
            
            return {
                'score': result.final_score,
                'label': result.final_label,
                'confidence': result.final_confidence,
                'agreement_rate': result.agreement_rate,
                'confidence_level': result.confidence_level,
                'position_recommendation': result.position_recommendation,
                'predictions': [p.to_dict() for p in result.individual_predictions],
                'total_latency_ms': result.total_latency_ms,
                'symbol': symbol,
                'ensemble': True,
            }
        except Exception as e:
            logger.error(f"Ensemble analysis failed: {e}")
            return self.analyze(symbol, text)
    
    def analyze_batch(
        self, 
        texts: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts efficiently.
        
        Args:
            texts: List of dicts with 'text', 'source', and optional 'symbol'
            
        Returns:
            List of analysis results
        """
        results = []
        for item in texts:
            text = item.get('text', '')
            source = item.get('source', 'unknown')
            symbol = item.get('symbol', 'BTCUSDT')
            
            result = self.analyze(symbol, text, source)
            result['original_text'] = text[:50]  # Truncate for logging
            results.append(result)
        
        return results
    
    def _get_btc_base_analyzer(self):
        """Get or create BTC base analyzer (fallback)."""
        if self._btc_base_analyzer is not None:
            return self._btc_base_analyzer
        
        try:
            from .hybrid_sentiment import HybridSentimentAnalyzer, HybridSentimentConfig
            
            config = HybridSentimentConfig(
                enable_lag_features=True,
                enable_dynamic_weighting=True
            )
            
            self._btc_base_analyzer = HybridSentimentAnalyzer(config)
            logger.info("BTC base analyzer initialized (fallback)")
            
            return self._btc_base_analyzer
            
        except Exception as e:
            logger.error(f"Failed to initialize BTC base analyzer: {e}")
            return None
    
    def _get_light_finetuned(self, symbol: str, config: AssetConfig):
        """Get or create light fine-tuned analyzer for asset."""
        if symbol in self._light_models:
            return self._light_models[symbol]
        
        self._enforce_memory_budget()
        
        # Fall back to multi-model or BTC base
        if self._multi_model_analyzer:
            return self._multi_model_analyzer
        return self._get_btc_base_analyzer()
    
    def _get_dedicated(self, symbol: str, config: AssetConfig):
        """Get or create dedicated analyzer for asset."""
        if symbol in self._dedicated_models:
            return self._dedicated_models[symbol]
        
        self._enforce_memory_budget()
        
        # Fall back to multi-model or BTC base
        if self._multi_model_analyzer:
            return self._multi_model_analyzer
        return self._get_btc_base_analyzer()
    
    def _enforce_memory_budget(self):
        """Evict least recently used models if over budget."""
        total_models = len(self._light_models) + len(self._dedicated_models)
        
        if total_models >= self.max_concurrent_models:
            all_symbols = list(self._light_models.keys()) + list(self._dedicated_models.keys())
            if not all_symbols:
                return
            
            lru_symbol = min(all_symbols, key=lambda s: self._model_usage.get(s, 0))
            
            if lru_symbol in self._light_models:
                del self._light_models[lru_symbol]
                logger.info(f"Evicted light model: {lru_symbol}")
            elif lru_symbol in self._dedicated_models:
                del self._dedicated_models[lru_symbol]
                logger.info(f"Evicted dedicated model: {lru_symbol}")
    
    def get_asset_config(self, symbol: str) -> Optional[AssetConfig]:
        """Get configuration for an asset."""
        return self.asset_configs.get(symbol)
    
    def register_asset(self, config: AssetConfig) -> None:
        """Register a new asset configuration."""
        self.asset_configs[config.symbol] = config
        logger.info(f"Registered asset: {config.symbol} ({config.model_variant.value})")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'multi_model_loaded': self._multi_model_analyzer is not None,
            'multi_model_phase': self.phase,
            'btc_base_loaded': self._btc_base_analyzer is not None,
            'light_models_count': len(self._light_models),
            'dedicated_models_count': len(self._dedicated_models),
            'total_models': (
                (1 if self._multi_model_analyzer else 0) +
                (1 if self._btc_base_analyzer else 0) +
                len(self._light_models) + 
                len(self._dedicated_models)
            ),
            'max_concurrent': self.max_concurrent_models,
        }
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of configured symbols."""
        return list(self.asset_configs.keys())
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics from multi-model analyzer."""
        if not self._multi_model_analyzer:
            return {}
        
        try:
            return self._multi_model_analyzer.get_metrics()
        except Exception:
            return {}
    
    def switch_phase(self, new_phase: int) -> bool:
        """
        Switch to a different multi-model phase.
        
        Args:
            new_phase: 1, 2, or 3
            
        Returns:
            True if switch successful
        """
        if new_phase not in [1, 2, 3]:
            logger.error(f"Invalid phase: {new_phase}")
            return False
        
        old_phase = self.phase
        self.phase = new_phase
        
        try:
            self._init_multi_model_analyzer()
            logger.info(f"Switched from Phase {old_phase} to Phase {new_phase}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch phase: {e}")
            self.phase = old_phase
            return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_sentiment_manager(phase: int = 2) -> MultiAssetSentimentManager:
    """Create sentiment manager with specified phase."""
    return MultiAssetSentimentManager(phase=phase)


def create_production_sentiment_manager() -> MultiAssetSentimentManager:
    """Create production-ready sentiment manager with Phase 2."""
    return MultiAssetSentimentManager(
        phase=2,
        max_concurrent_models=5,
    )
