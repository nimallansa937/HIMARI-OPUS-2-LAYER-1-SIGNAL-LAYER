"""
Multi-Model Sentiment Ensemble for HIMARI
==========================================

Implements the HIMARI Sentiment Model Integration Guide (v2.0).

Phase 1: CryptoBERT + ModernFinBERT
Phase 2: FinTwitBERT + DeBERTa-v3 ensemble
Phase 3: DistilRoBERTa fallback

Architecture:
    SOCIAL MEDIA (Twitter, Reddit, Telegram, StockTwits)
        │
        └──► CryptoBERT (70%, 8-12ms) ──► ALERT signal
        
    NEWS (Bloomberg, Reuters, CoinDesk)
        │
        ├──► Your fine-tuned RoBERTa (85%, 40ms) ──► TRADE signal (high confidence)
        │
        └──► ModernFinBERT (75%, 5-8ms) ──► TRADE signal (speed priority)

    ENSEMBLE VOTING (Phase 2+)
        │
        ├──► 3/3 agree: HIGH confidence → Full position
        ├──► 2/3 agree: MEDIUM confidence → Reduced position
        └──► Diverge: LOW confidence → HOLD or manual review

Expected Sharpe Improvement: +0.8 to +1.8
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelType(Enum):
    """Supported model types."""
    CRYPTOBERT = "cryptobert"
    MODERNFINBERT = "modernfinbert"
    FINTWITBERT = "fintwitbert"
    DEBERTA = "deberta"
    DISTILROBERTA = "distilroberta"
    FINETUNED_ROBERTA = "finetuned_roberta"
    TWITTER_ROBERTA = "twitter_roberta"


@dataclass
class ModelConfig:
    """Configuration for a single sentiment model."""
    name: str
    huggingface_id: str
    model_type: ModelType
    latency_target_ms: float
    expected_accuracy: float
    is_crypto_specific: bool = False
    is_fast_path: bool = True
    monthly_cost_usd: float = 0.0
    max_length: int = 128
    

# Model registry with HuggingFace IDs and configurations
MODEL_REGISTRY: Dict[ModelType, ModelConfig] = {
    ModelType.CRYPTOBERT: ModelConfig(
        name="CryptoBERT",
        huggingface_id="ElKulako/cryptobert",
        model_type=ModelType.CRYPTOBERT,
        latency_target_ms=12.0,
        expected_accuracy=0.70,
        is_crypto_specific=True,
        is_fast_path=True,
        monthly_cost_usd=5.0,
    ),
    ModelType.MODERNFINBERT: ModelConfig(
        name="ModernFinBERT",
        huggingface_id="tabularisai/ModernFinBERT",
        model_type=ModelType.MODERNFINBERT,
        latency_target_ms=8.0,
        expected_accuracy=0.75,
        is_crypto_specific=False,
        is_fast_path=True,
        monthly_cost_usd=6.0,
    ),
    ModelType.FINTWITBERT: ModelConfig(
        name="FinTwitBERT",
        huggingface_id="StephanAkkerman/FinTwitBERT-sentiment",
        model_type=ModelType.FINTWITBERT,
        latency_target_ms=10.0,
        expected_accuracy=0.75,
        is_crypto_specific=False,
        is_fast_path=True,
        monthly_cost_usd=5.0,
    ),
    ModelType.DEBERTA: ModelConfig(
        name="DeBERTa-v3 Finance",
        huggingface_id="nickmuchi/deberta-v3-base-finetuned-finance-text-classification",
        model_type=ModelType.DEBERTA,
        latency_target_ms=10.0,
        expected_accuracy=0.89,
        is_crypto_specific=False,
        is_fast_path=False,
        monthly_cost_usd=5.0,
        max_length=256,
    ),
    ModelType.DISTILROBERTA: ModelConfig(
        name="DistilRoBERTa Financial",
        huggingface_id="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        model_type=ModelType.DISTILROBERTA,
        latency_target_ms=4.0,
        expected_accuracy=0.75,
        is_crypto_specific=False,
        is_fast_path=True,
        monthly_cost_usd=2.0,
    ),
    ModelType.TWITTER_ROBERTA: ModelConfig(
        name="Twitter-RoBERTa",
        huggingface_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_type=ModelType.TWITTER_ROBERTA,
        latency_target_ms=5.0,
        expected_accuracy=0.72,
        is_crypto_specific=False,
        is_fast_path=True,
        monthly_cost_usd=0.0,  # Already included
    ),
}


# Unified label mapping: All models output LABEL_0, LABEL_1, LABEL_2
# Map to -1 (bearish), 0 (neutral), +1 (bullish)
LABEL_MAPS: Dict[ModelType, Dict[int, int]] = {
    ModelType.CRYPTOBERT:      {0: -1, 1: 0, 2: +1},  # Bearish/Neutral/Bullish
    ModelType.MODERNFINBERT:   {0: -1, 1: 0, 2: +1},  # Negative/Neutral/Positive
    ModelType.FINTWITBERT:     {0: -1, 1: 0, 2: +1},  # Negative/Neutral/Positive
    ModelType.DEBERTA:         {0: -1, 1: 0, 2: +1},  # Negative/Neutral/Positive
    ModelType.DISTILROBERTA:   {0: -1, 1: 0, 2: +1},  # Negative/Neutral/Positive
    ModelType.TWITTER_ROBERTA: {0: -1, 1: 0, 2: +1},  # Negative/Neutral/Positive
    ModelType.FINETUNED_ROBERTA: {0: -1, 1: 0, 2: +1},
}


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for the multi-model ensemble."""
    
    # Which models to load (Phase 1, 2, or 3)
    phase: int = 1
    
    # Primary models for each path
    social_primary: ModelType = ModelType.CRYPTOBERT
    news_primary: ModelType = ModelType.MODERNFINBERT
    
    # Ensemble members for voting (Phase 2+)
    ensemble_members: List[ModelType] = field(default_factory=lambda: [
        ModelType.CRYPTOBERT,
        ModelType.MODERNFINBERT,
        ModelType.FINTWITBERT,
    ])
    
    # Fallback model (Phase 3)
    fallback_model: ModelType = ModelType.DISTILROBERTA
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.60
    
    # Ensemble agreement thresholds
    full_agreement_threshold: float = 1.0    # 3/3 agree
    majority_agreement_threshold: float = 0.67  # 2/3 agree
    
    # Performance thresholds
    max_latency_ms: float = 20.0
    max_requests_per_sec: int = 1000  # Switch to fallback above this
    
    # Device
    device: str = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"


# =============================================================================
# MODEL RESULT
# =============================================================================

@dataclass
class ModelPrediction:
    """Single model prediction result."""
    model_type: ModelType
    model_name: str
    score: float           # -1 to +1
    confidence: float      # 0 to 1
    label: str             # "bullish", "bearish", "neutral"
    label_index: int       # Raw label index
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EnsembleResult:
    """Ensemble voting result."""
    final_score: float
    final_confidence: float
    final_label: str
    agreement_rate: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    position_recommendation: str  # "FULL", "REDUCED", "HOLD"
    individual_predictions: List[ModelPrediction]
    total_latency_ms: float
    fallback_used: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "score": self.final_score,
            "confidence": self.final_confidence,
            "label": self.final_label,
            "agreement_rate": self.agreement_rate,
            "confidence_level": self.confidence_level,
            "position_recommendation": self.position_recommendation,
            "predictions": [p.to_dict() for p in self.individual_predictions],
            "total_latency_ms": self.total_latency_ms,
            "fallback_used": self.fallback_used,
        }


# =============================================================================
# MULTI-MODEL SENTIMENT ANALYZER
# =============================================================================

class MultiModelSentimentAnalyzer:
    """
    Multi-model sentiment analyzer with ensemble voting.
    
    Implements HIMARI Sentiment Model Integration Guide v2.0:
    - Phase 1: CryptoBERT + ModernFinBERT
    - Phase 2: + FinTwitBERT + DeBERTa ensemble
    - Phase 3: + DistilRoBERTa fallback
    
    Example:
        analyzer = MultiModelSentimentAnalyzer(phase=1)
        
        # Social media → CryptoBERT
        result = analyzer.analyze("BTC mooning! 🚀", source="twitter")
        
        # News → ModernFinBERT
        result = analyzer.analyze("Bitcoin rallies 10%", source="bloomberg")
        
        # Full ensemble (Phase 2+)
        result = analyzer.analyze_ensemble("Breaking: ETF approved!")
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # Loaded models
        self._models: Dict[ModelType, Any] = {}
        self._tokenizers: Dict[ModelType, Any] = {}
        
        # Metrics
        self._call_counts: Dict[ModelType, int] = {}
        self._latencies: Dict[ModelType, List[float]] = {}
        self._agreement_history: List[float] = []
        
        # Load models based on phase
        self._load_models()
        
        logger.info(f"MultiModelSentimentAnalyzer initialized (Phase {self.config.phase})")
    
    def _load_models(self) -> None:
        """Load models based on configured phase."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Install: pip install transformers torch")
            return
        
        models_to_load = []
        
        # Phase 1: Primary models only
        if self.config.phase >= 1:
            models_to_load.extend([
                self.config.social_primary,
                self.config.news_primary,
            ])
        
        # Phase 2: Add ensemble members
        if self.config.phase >= 2:
            for model_type in self.config.ensemble_members:
                if model_type not in models_to_load:
                    models_to_load.append(model_type)
        
        # Phase 3: Add fallback
        if self.config.phase >= 3:
            if self.config.fallback_model not in models_to_load:
                models_to_load.append(self.config.fallback_model)
        
        # Load each model
        for model_type in models_to_load:
            self._load_single_model(model_type)
    
    def _load_single_model(self, model_type: ModelType) -> bool:
        """Load a single model from HuggingFace."""
        if model_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown model type: {model_type}")
            return False
        
        config = MODEL_REGISTRY[model_type]
        
        try:
            logger.info(f"Loading {config.name}: {config.huggingface_id}")
            
            # Use pipeline for simplicity
            model_pipeline = pipeline(
                "sentiment-analysis",
                model=config.huggingface_id,
                tokenizer=config.huggingface_id,
                device=0 if self.config.device == "cuda" else -1,
                truncation=True,
                max_length=config.max_length,
            )
            
            self._models[model_type] = model_pipeline
            self._call_counts[model_type] = 0
            self._latencies[model_type] = []
            
            logger.info(f"✓ {config.name} loaded on {self.config.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            return False
    
    def _predict_single(
        self, 
        text: str, 
        model_type: ModelType
    ) -> Optional[ModelPrediction]:
        """Run prediction with a single model."""
        if model_type not in self._models:
            return None
        
        config = MODEL_REGISTRY[model_type]
        model = self._models[model_type]
        
        try:
            start_time = time.perf_counter()
            
            # Run inference
            result = model(text)[0]
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse result
            raw_label = result['label']
            confidence = result['score']
            
            # Extract label index from LABEL_X format
            if raw_label.startswith('LABEL_'):
                label_index = int(raw_label.split('_')[1])
            else:
                # Try to parse directly
                label_index = 1  # Default neutral
                if 'positive' in raw_label.lower() or 'bullish' in raw_label.lower():
                    label_index = 2
                elif 'negative' in raw_label.lower() or 'bearish' in raw_label.lower():
                    label_index = 0
            
            # Map to unified score
            label_map = LABEL_MAPS.get(model_type, {0: -1, 1: 0, 2: +1})
            sentiment_direction = label_map.get(label_index, 0)
            
            # Score = direction * confidence
            score = sentiment_direction * confidence
            
            # Label text
            if sentiment_direction > 0:
                label = "bullish"
            elif sentiment_direction < 0:
                label = "bearish"
            else:
                label = "neutral"
            
            # Track metrics
            self._call_counts[model_type] += 1
            self._latencies[model_type].append(latency_ms)
            
            return ModelPrediction(
                model_type=model_type,
                model_name=config.name,
                score=score,
                confidence=confidence,
                label=label,
                label_index=label_index,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {config.name}: {e}")
            return None
    
    def analyze(
        self, 
        text: str, 
        source: str = "unknown"
    ) -> Optional[ModelPrediction]:
        """
        Analyze text with appropriate model based on source.
        
        Routes:
        - Social sources → CryptoBERT (or social_primary)
        - News sources → ModernFinBERT (or news_primary)
        
        Args:
            text: Input text
            source: Source type (twitter, reddit, bloomberg, etc.)
            
        Returns:
            ModelPrediction from the appropriate model
        """
        # Determine which model to use
        social_sources = {'twitter', 'reddit', 'telegram', 'discord', 'stocktwits'}
        news_sources = {'bloomberg', 'reuters', 'coindesk', 'cointelegraph', 'news'}
        
        source_lower = source.lower()
        
        if source_lower in social_sources:
            model_type = self.config.social_primary
        elif source_lower in news_sources:
            model_type = self.config.news_primary
        else:
            # Default to social primary for speed
            model_type = self.config.social_primary
        
        return self._predict_single(text, model_type)
    
    def analyze_ensemble(
        self, 
        text: str,
        models: Optional[List[ModelType]] = None
    ) -> EnsembleResult:
        """
        Analyze text with ensemble voting (Phase 2+).
        
        Voting logic:
        - 3/3 agree → HIGH confidence → Full position
        - 2/3 agree → MEDIUM confidence → Reduced position
        - Diverge → LOW confidence → HOLD
        
        Args:
            text: Input text
            models: Optional list of models to use (default: ensemble_members)
            
        Returns:
            EnsembleResult with voting details
        """
        models_to_use = models or self.config.ensemble_members
        
        predictions: List[ModelPrediction] = []
        total_latency = 0.0
        
        for model_type in models_to_use:
            pred = self._predict_single(text, model_type)
            if pred:
                predictions.append(pred)
                total_latency += pred.latency_ms
        
        if not predictions:
            # Use fallback if available
            if self.config.phase >= 3 and self.config.fallback_model in self._models:
                fallback_pred = self._predict_single(text, self.config.fallback_model)
                if fallback_pred:
                    return EnsembleResult(
                        final_score=fallback_pred.score,
                        final_confidence=fallback_pred.confidence,
                        final_label=fallback_pred.label,
                        agreement_rate=1.0,
                        confidence_level="LOW",
                        position_recommendation="REDUCED",
                        individual_predictions=[fallback_pred],
                        total_latency_ms=fallback_pred.latency_ms,
                        fallback_used=True,
                    )
            
            # No predictions available
            return EnsembleResult(
                final_score=0.0,
                final_confidence=0.0,
                final_label="neutral",
                agreement_rate=0.0,
                confidence_level="LOW",
                position_recommendation="HOLD",
                individual_predictions=[],
                total_latency_ms=0.0,
                fallback_used=False,
            )
        
        # Compute ensemble vote
        scores = [p.score for p in predictions]
        confidences = [p.confidence for p in predictions]
        labels = [p.label for p in predictions]
        
        # Agreement rate (how many agree with majority)
        from collections import Counter
        label_counts = Counter(labels)
        majority_label, majority_count = label_counts.most_common(1)[0]
        agreement_rate = majority_count / len(predictions)
        
        # Track agreement
        self._agreement_history.append(agreement_rate)
        if len(self._agreement_history) > 1000:
            self._agreement_history = self._agreement_history[-1000:]
        
        # Final score: weighted average by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            final_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
        else:
            final_score = np.mean(scores)
        
        final_confidence = np.mean(confidences)
        
        # Determine confidence level and position recommendation
        if agreement_rate >= self.config.full_agreement_threshold:
            confidence_level = "HIGH"
            position_recommendation = "FULL"
        elif agreement_rate >= self.config.majority_agreement_threshold:
            confidence_level = "MEDIUM"
            position_recommendation = "REDUCED"
        else:
            confidence_level = "LOW"
            position_recommendation = "HOLD"
        
        # Final label
        if final_score > 0.3:
            final_label = "bullish"
        elif final_score < -0.3:
            final_label = "bearish"
        else:
            final_label = "neutral"
        
        return EnsembleResult(
            final_score=final_score,
            final_confidence=final_confidence,
            final_label=final_label,
            agreement_rate=agreement_rate,
            confidence_level=confidence_level,
            position_recommendation=position_recommendation,
            individual_predictions=predictions,
            total_latency_ms=total_latency,
            fallback_used=False,
        )
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for all loaded models."""
        metrics = {
            "phase": self.config.phase,
            "models_loaded": list(self._models.keys()),
            "agreement_rate_avg": np.mean(self._agreement_history) if self._agreement_history else 0.0,
            "per_model": {},
        }
        
        for model_type, count in self._call_counts.items():
            latencies = self._latencies.get(model_type, [])
            config = MODEL_REGISTRY[model_type]
            
            metrics["per_model"][model_type.value] = {
                "name": config.name,
                "calls": count,
                "latency_p50_ms": np.percentile(latencies[-1000:], 50) if latencies else 0.0,
                "latency_p99_ms": np.percentile(latencies[-1000:], 99) if latencies else 0.0,
                "target_ms": config.latency_target_ms,
                "expected_accuracy": config.expected_accuracy,
            }
        
        return metrics


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_phase1_analyzer(device: str = "auto") -> MultiModelSentimentAnalyzer:
    """Create Phase 1 analyzer with CryptoBERT + ModernFinBERT."""
    if device == "auto":
        device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    config = EnsembleConfig(
        phase=1,
        social_primary=ModelType.CRYPTOBERT,
        news_primary=ModelType.MODERNFINBERT,
        device=device,
    )
    
    return MultiModelSentimentAnalyzer(config)


def create_phase2_analyzer(device: str = "auto") -> MultiModelSentimentAnalyzer:
    """Create Phase 2 analyzer with full ensemble."""
    if device == "auto":
        device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    config = EnsembleConfig(
        phase=2,
        social_primary=ModelType.CRYPTOBERT,
        news_primary=ModelType.MODERNFINBERT,
        ensemble_members=[
            ModelType.CRYPTOBERT,
            ModelType.MODERNFINBERT,
            ModelType.FINTWITBERT,
        ],
        device=device,
    )
    
    return MultiModelSentimentAnalyzer(config)


def create_phase3_analyzer(device: str = "auto") -> MultiModelSentimentAnalyzer:
    """Create Phase 3 analyzer with fallback."""
    if device == "auto":
        device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    config = EnsembleConfig(
        phase=3,
        social_primary=ModelType.CRYPTOBERT,
        news_primary=ModelType.MODERNFINBERT,
        ensemble_members=[
            ModelType.CRYPTOBERT,
            ModelType.MODERNFINBERT,
            ModelType.FINTWITBERT,
        ],
        fallback_model=ModelType.DISTILROBERTA,
        device=device,
    )
    
    return MultiModelSentimentAnalyzer(config)


# =============================================================================
# QUICK VALIDATION TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("MULTI-MODEL SENTIMENT ANALYZER VALIDATION")
    print("=" * 60)
    
    # Test Phase 1
    print("\n📦 Loading Phase 1 models...")
    analyzer = create_phase1_analyzer()
    
    # Test cases
    tests = [
        ("Bitcoin breaking $50k! 🚀", "twitter", "bullish"),
        ("SEC crackdown incoming 📉", "twitter", "bearish"),
        ("HODL this dumpster fire 💀", "twitter", "bearish"),
        ("Bitcoin rallies as institutional demand grows", "bloomberg", "bullish"),
        ("Crypto market faces regulatory headwinds", "reuters", "bearish"),
    ]
    
    print("\n🧪 Running test cases...\n")
    
    for text, source, expected in tests:
        result = analyzer.analyze(text, source)
        if result:
            match = "✓" if result.label == expected else "✗"
            print(f"{match} [{source:10}] {text[:40]}...")
            print(f"   Model: {result.model_name}")
            print(f"   Score: {result.score:+.3f} | Label: {result.label} | Expected: {expected}")
            print(f"   Latency: {result.latency_ms:.1f}ms | Confidence: {result.confidence:.2%}")
            print()
    
    # Print metrics
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    metrics = analyzer.get_metrics()
    print(f"Models loaded: {len(metrics['models_loaded'])}")
    for model_name, stats in metrics['per_model'].items():
        print(f"\n{stats['name']}:")
        print(f"  Calls: {stats['calls']}")
        print(f"  Latency p50: {stats['latency_p50_ms']:.1f}ms (target: {stats['target_ms']:.0f}ms)")
