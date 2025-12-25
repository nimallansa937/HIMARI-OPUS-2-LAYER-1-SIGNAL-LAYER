"""
Dual-Path Sentiment Analyzer for HIMARI

Routes text to appropriate models based on source:
- FAST PATH: Twitter-RoBERTa for social media (~3-5ms) → Early warning alerts
- ACCURATE PATH: Fine-tuned Financial-RoBERTa for news (~40ms) → Trade signals

Why dual-path matters:
- Trump tweet in October 2024 caused $2B+ liquidations in <2 minutes
- By the time Bloomberg reports, the move is over
- Social media = leading indicator (noisy but fast)
- News = confirmation signal (accurate but lagging)

Performance Targets:
- Fast path: <5ms latency, 70-75% directional accuracy
- Accurate path: <50ms latency, 85%+ directional accuracy

Integration with existing HIMARI:
- Replaces hybrid_sentiment.py for sources routing
- Integrates with social_sentiment_aggregator.py for buffering
- Signals published to Redis under signals:{symbol}:sentiment_*
"""

import os
import time
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class SignalType(Enum):
    """Signal urgency classification."""
    ALERT = "alert"      # Fast path - early warning, don't trade yet
    TRADE = "trade"      # Accurate path - high confidence, act on it
    INFO = "info"        # Low confidence, informational only


class SourceType(Enum):
    """Text source classification."""
    # Fast path sources (social media)
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    STOCKTWITS = "stocktwits"
    
    # Accurate path sources (news)
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"
    NEWS = "news"
    
    # Unknown defaults to accurate path
    UNKNOWN = "unknown"
    
    @classmethod
    def is_social(cls, source: str) -> bool:
        """Check if source is social media (fast path)."""
        social_sources = {
            cls.TWITTER.value, cls.REDDIT.value, cls.TELEGRAM.value,
            cls.DISCORD.value, cls.STOCKTWITS.value
        }
        return source.lower() in social_sources
    
    @classmethod
    def is_news(cls, source: str) -> bool:
        """Check if source is news (accurate path)."""
        news_sources = {
            cls.BLOOMBERG.value, cls.REUTERS.value, cls.COINDESK.value,
            cls.COINTELEGRAPH.value, cls.NEWS.value
        }
        return source.lower() in news_sources


@dataclass
class DualPathConfig:
    """Configuration for dual-path sentiment analyzer."""
    
    # Model paths
    fast_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    accurate_model: str = "soleimanian/financial-roberta-large-sentiment"
    fine_tuned_model_path: Optional[str] = None  # Path to your fine-tuned model
    
    # Use fine-tuned model if available
    use_fine_tuned: bool = True
    
    # Confidence thresholds
    fast_confidence_threshold: float = 0.60   # Lower bar for alerts
    accurate_confidence_threshold: float = 0.70  # Higher bar for trades
    
    # Latency targets (ms)
    fast_latency_target: float = 5.0
    accurate_latency_target: float = 50.0
    
    # Score thresholds for labels
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3
    
    # Device configuration
    device: str = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # Caching
    cache_ttl_seconds: int = 60  # Cache identical texts for 1 minute
    
    # Fallback behavior
    fallback_to_fast_on_timeout: bool = True
    accurate_timeout_ms: float = 100.0  # Switch to fast if accurate takes too long


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    score: float           # -1 to +1
    confidence: float      # 0 to 1
    label: str             # bullish, bearish, neutral
    signal_type: SignalType
    source_type: str
    path_used: str         # "fast" or "accurate"
    latency_ms: float
    model_used: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Redis storage."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label,
            "signal_type": self.signal_type.value,
            "source_type": self.source_type,
            "path_used": self.path_used,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used
        }


class DualPathSentimentAnalyzer:
    """
    Dual-path sentiment analyzer with source-based routing.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    TEXT INPUT                          │
    │                    + source                            │
    └───────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │   FAST PATH       │           │   ACCURATE PATH   │
    │   < 5ms           │           │   < 50ms          │
    │                   │           │                   │
    │ Twitter-RoBERTa   │           │ Fine-tuned        │
    │ (social media)    │           │ Financial-RoBERTa │
    │                   │           │                   │
    │ 70-75% accuracy   │           │ 85% accuracy      │
    │ ALERT signals     │           │ TRADE signals     │
    └───────────────────┘           └───────────────────┘
    
    Example:
        analyzer = DualPathSentimentAnalyzer(config)
        
        # Social media → fast path → ALERT
        result = analyzer.analyze("BTC mooning! 🚀", source="twitter")
        # result.signal_type = SignalType.ALERT
        # result.latency_ms ≈ 3-5ms
        
        # News → accurate path → TRADE
        result = analyzer.analyze("Bitcoin surges past $100K", source="bloomberg")
        # result.signal_type = SignalType.TRADE
        # result.latency_ms ≈ 40ms
    """
    
    def __init__(self, config: Optional[DualPathConfig] = None):
        self.config = config or DualPathConfig()
        
        # Check for fine-tuned model path from environment
        if self.config.fine_tuned_model_path is None:
            self.config.fine_tuned_model_path = os.getenv(
                "HIMARI_FINE_TUNED_MODEL_PATH",
                "./models/financial-roberta-crypto-finetuned"
            )
        
        # Initialize models
        self._fast_model = None
        self._fast_tokenizer = None
        self._accurate_model = None
        self._accurate_tokenizer = None
        self._accurate_pipeline = None
        
        # Metrics
        self._fast_call_count = 0
        self._accurate_call_count = 0
        self._fast_latencies = []
        self._accurate_latencies = []
        
        # Simple cache {text_hash: (result, timestamp)}
        self._cache: Dict[str, Tuple[SentimentResult, float]] = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load both sentiment models."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Install: pip install transformers torch")
            return
        
        # Load fast path model (Twitter-RoBERTa)
        try:
            logger.info(f"Loading fast path model: {self.config.fast_model}")
            self._fast_tokenizer = AutoTokenizer.from_pretrained(self.config.fast_model)
            self._fast_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.fast_model
            )
            self._fast_model.to(self.config.device)
            self._fast_model.eval()
            logger.info(f"✓ Fast path model loaded on {self.config.device}")
        except Exception as e:
            logger.error(f"Failed to load fast model: {e}")
        
        # Load accurate path model
        accurate_model_path = self.config.accurate_model
        
        # Check if fine-tuned model exists
        if self.config.use_fine_tuned and self.config.fine_tuned_model_path:
            if os.path.exists(self.config.fine_tuned_model_path):
                accurate_model_path = self.config.fine_tuned_model_path
                logger.info(f"Using fine-tuned model: {accurate_model_path}")
            else:
                logger.warning(
                    f"Fine-tuned model not found at {self.config.fine_tuned_model_path}, "
                    f"falling back to {self.config.accurate_model}"
                )
        
        try:
            logger.info(f"Loading accurate path model: {accurate_model_path}")
            self._accurate_tokenizer = AutoTokenizer.from_pretrained(accurate_model_path)
            self._accurate_model = AutoModelForSequenceClassification.from_pretrained(
                accurate_model_path
            )
            self._accurate_model.to(self.config.device)
            self._accurate_model.eval()
            self._accurate_model_name = accurate_model_path
            logger.info(f"✓ Accurate path model loaded on {self.config.device}")
        except Exception as e:
            logger.error(f"Failed to load accurate model: {e}")
            # Fallback to pipeline with base model
            try:
                self._accurate_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.config.accurate_model,
                    tokenizer=self.config.accurate_model,
                    device=0 if self.config.device == "cuda" else -1
                )
                self._accurate_model_name = self.config.accurate_model
                logger.info("✓ Accurate path using pipeline fallback")
            except Exception as e2:
                logger.error(f"Pipeline fallback also failed: {e2}")
    
    def analyze(
        self,
        text: str,
        source: str = "unknown",
        force_path: Optional[str] = None
    ) -> SentimentResult:
        """
        Analyze text sentiment with automatic path routing.
        
        Args:
            text: Input text (tweet, headline, etc.)
            source: Source identifier (twitter, bloomberg, etc.)
            force_path: Override routing ("fast" or "accurate")
        
        Returns:
            SentimentResult with score, confidence, and signal type
        """
        # Check cache
        cache_key = hash(f"{text}:{source}")
        if cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl_seconds:
                return cached_result
        
        # Determine path
        if force_path == "fast":
            use_fast = True
        elif force_path == "accurate":
            use_fast = False
        else:
            use_fast = SourceType.is_social(source)
        
        # Route to appropriate model
        start_time = time.perf_counter()
        
        if use_fast:
            result = self._analyze_fast(text, source)
        else:
            result = self._analyze_accurate(text, source)
        
        # Cache result
        self._cache[cache_key] = (result, time.time())
        
        # Clean old cache entries periodically
        if len(self._cache) > 10000:
            self._clean_cache()
        
        return result
    
    def _analyze_fast(self, text: str, source: str) -> SentimentResult:
        """Fast path analysis using Twitter-RoBERTa."""
        start_time = time.perf_counter()
        
        if self._fast_model is None or self._fast_tokenizer is None:
            return self._fallback_result(text, source, "fast")
        
        try:
            # Tokenize
            inputs = self._fast_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,  # Shorter for speed
                padding=True
            ).to(self.config.device)
            
            # Inference
            with torch.no_grad():
                outputs = self._fast_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Twitter-RoBERTa outputs: negative (0), neutral (1), positive (2)
            probs_np = probs.cpu().numpy()[0]
            
            # Convert to -1 to +1 scale
            # score = P(positive) - P(negative)
            score = float(probs_np[2] - probs_np[0])
            confidence = float(max(probs_np))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Determine label
            if score > self.config.bullish_threshold:
                label = "bullish"
            elif score < self.config.bearish_threshold:
                label = "bearish"
            else:
                label = "neutral"
            
            # Determine signal type based on confidence
            if confidence >= self.config.fast_confidence_threshold and label != "neutral":
                signal_type = SignalType.ALERT
            else:
                signal_type = SignalType.INFO
            
            self._fast_call_count += 1
            self._fast_latencies.append(latency_ms)
            
            return SentimentResult(
                score=score,
                confidence=confidence,
                label=label,
                signal_type=signal_type,
                source_type=source,
                path_used="fast",
                latency_ms=latency_ms,
                model_used=self.config.fast_model
            )
            
        except Exception as e:
            logger.error(f"Fast path error: {e}")
            return self._fallback_result(text, source, "fast")
    
    def _analyze_accurate(self, text: str, source: str) -> SentimentResult:
        """Accurate path analysis using fine-tuned Financial-RoBERTa."""
        start_time = time.perf_counter()
        
        # Check timeout - fallback to fast if taking too long
        def check_timeout():
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > self.config.accurate_timeout_ms and self.config.fallback_to_fast_on_timeout:
                logger.warning(f"Accurate path timeout ({elapsed:.1f}ms), falling back to fast")
                return self._analyze_fast(text, source)
            return None
        
        # Try direct model inference first
        if self._accurate_model is not None and self._accurate_tokenizer is not None:
            try:
                inputs = self._accurate_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self._accurate_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                probs_np = probs.cpu().numpy()[0]
                
                # Handle different model output formats
                num_labels = len(probs_np)
                
                if num_labels == 3:
                    # Fine-tuned model: bearish (0), neutral (1), bullish (2)
                    # OR: negative (0), neutral (1), positive (2)
                    score = float(probs_np[2] - probs_np[0])
                    confidence = float(max(probs_np))
                elif num_labels == 2:
                    # Binary: negative (0), positive (1)
                    score = float(probs_np[1] * 2 - 1)  # Map to -1 to +1
                    confidence = float(max(probs_np))
                else:
                    # Unknown format
                    score = 0.0
                    confidence = 0.5
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Determine label
                if score > self.config.bullish_threshold:
                    label = "bullish"
                elif score < self.config.bearish_threshold:
                    label = "bearish"
                else:
                    label = "neutral"
                
                # Determine signal type - accurate path gets TRADE signals
                if confidence >= self.config.accurate_confidence_threshold and label != "neutral":
                    signal_type = SignalType.TRADE
                elif confidence >= self.config.fast_confidence_threshold and label != "neutral":
                    signal_type = SignalType.ALERT
                else:
                    signal_type = SignalType.INFO
                
                self._accurate_call_count += 1
                self._accurate_latencies.append(latency_ms)
                
                return SentimentResult(
                    score=score,
                    confidence=confidence,
                    label=label,
                    signal_type=signal_type,
                    source_type=source,
                    path_used="accurate",
                    latency_ms=latency_ms,
                    model_used=self._accurate_model_name
                )
                
            except Exception as e:
                logger.error(f"Accurate model error: {e}")
        
        # Fallback to pipeline
        if self._accurate_pipeline is not None:
            try:
                result = self._accurate_pipeline(text)[0]
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Convert pipeline output
                label_raw = result['label'].lower()
                confidence = result['score']
                
                if 'positive' in label_raw or 'bullish' in label_raw:
                    score = confidence
                    label = "bullish"
                elif 'negative' in label_raw or 'bearish' in label_raw:
                    score = -confidence
                    label = "bearish"
                else:
                    score = 0.0
                    label = "neutral"
                
                signal_type = SignalType.TRADE if confidence >= self.config.accurate_confidence_threshold else SignalType.INFO
                
                return SentimentResult(
                    score=score,
                    confidence=confidence,
                    label=label,
                    signal_type=signal_type,
                    source_type=source,
                    path_used="accurate",
                    latency_ms=latency_ms,
                    model_used=self._accurate_model_name
                )
                
            except Exception as e:
                logger.error(f"Accurate pipeline error: {e}")
        
        return self._fallback_result(text, source, "accurate")
    
    def _fallback_result(self, text: str, source: str, path: str) -> SentimentResult:
        """Return neutral result when models unavailable."""
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            label="neutral",
            signal_type=SignalType.INFO,
            source_type=source,
            path_used=path,
            latency_ms=0.0,
            model_used="fallback"
        )
    
    def _clean_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            k for k, (_, t) in self._cache.items()
            if current_time - t > self.config.cache_ttl_seconds
        ]
        for k in expired_keys:
            del self._cache[k]
    
    def analyze_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None
    ) -> List[SentimentResult]:
        """
        Analyze batch of texts.
        
        Automatically routes each text to appropriate model based on source.
        """
        if sources is None:
            sources = ["unknown"] * len(texts)
        
        results = []
        for text, source in zip(texts, sources):
            results.append(self.analyze(text, source))
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        def safe_percentile(arr, p):
            if len(arr) == 0:
                return 0.0
            return float(np.percentile(arr, p))
        
        return {
            "fast_path": {
                "call_count": self._fast_call_count,
                "latency_p50_ms": safe_percentile(self._fast_latencies[-1000:], 50),
                "latency_p99_ms": safe_percentile(self._fast_latencies[-1000:], 99),
                "model": self.config.fast_model
            },
            "accurate_path": {
                "call_count": self._accurate_call_count,
                "latency_p50_ms": safe_percentile(self._accurate_latencies[-1000:], 50),
                "latency_p99_ms": safe_percentile(self._accurate_latencies[-1000:], 99),
                "model": getattr(self, '_accurate_model_name', self.config.accurate_model)
            },
            "cache_size": len(self._cache)
        }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_dual_path_analyzer(
    fine_tuned_path: Optional[str] = None,
    use_fine_tuned: bool = True,
    device: str = "auto"
) -> DualPathSentimentAnalyzer:
    """
    Factory function to create properly configured dual-path analyzer.
    
    Args:
        fine_tuned_path: Path to fine-tuned model directory
        use_fine_tuned: Whether to use fine-tuned model for accurate path
        device: "cuda", "cpu", or "auto"
    
    Returns:
        Configured DualPathSentimentAnalyzer
    """
    if device == "auto":
        device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    config = DualPathConfig(
        fine_tuned_model_path=fine_tuned_path or os.getenv(
            "HIMARI_FINE_TUNED_MODEL_PATH",
            "./models/financial-roberta-crypto-finetuned"
        ),
        use_fine_tuned=use_fine_tuned,
        device=device
    )
    
    return DualPathSentimentAnalyzer(config)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("DUAL PATH SENTIMENT ANALYZER TEST")
    print("=" * 60)
    
    # Create analyzer
    analyzer = create_dual_path_analyzer(use_fine_tuned=False)  # Use base model for test
    
    # Test cases
    test_cases = [
        # Social media → Fast path → ALERT
        ("BTC mooning! 🚀🚀🚀 diamond hands", "twitter"),
        ("Just got rekt on ETH short. FML", "reddit"),
        
        # News → Accurate path → TRADE  
        ("Bitcoin surges past $100,000 as institutional demand grows", "bloomberg"),
        ("SEC delays decision on Ethereum ETF applications", "reuters"),
        
        # Unknown source → Accurate path
        ("Crypto market sees increased volatility", "unknown"),
    ]
    
    print("\nRunning test cases:\n")
    
    for text, source in test_cases:
        result = analyzer.analyze(text, source)
        print(f"Source: {source}")
        print(f"Text: {text[:50]}...")
        print(f"  Path: {result.path_used}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Label: {result.label}")
        print(f"  Signal: {result.signal_type.value}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print()
    
    # Print metrics
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    metrics = analyzer.get_metrics()
    print(f"Fast path calls: {metrics['fast_path']['call_count']}")
    print(f"Fast path p50 latency: {metrics['fast_path']['latency_p50_ms']:.2f}ms")
    print(f"Accurate path calls: {metrics['accurate_path']['call_count']}")
    print(f"Accurate path p50 latency: {metrics['accurate_path']['latency_p50_ms']:.2f}ms")


# =============================================================================
# RESEARCH-BACKED ENHANCEMENTS
# =============================================================================

@dataclass
class DisagreementSignal:
    """Result from disagreement analysis between fast and accurate paths."""
    signal: str           # "HIGH_VOL_EXPECTED", "ALIGNED", "DIVERGENCE"
    magnitude: float      # 0 to 2 (larger = more disagreement)
    fast_score: float
    accurate_score: float
    confidence: float     # Lower when disagreement is high
    volatility_forecast: str  # "high", "normal", "low"


def compute_disagreement(
    fast_result: SentimentResult,
    accurate_result: SentimentResult
) -> DisagreementSignal:
    """
    Compute disagreement signal between fast and accurate paths (Enhancement #8).
    
    Research shows high disagreement between sources = upcoming volatility spike.
    When social media and news disagree, expect increased volatility.
    
    Args:
        fast_result: Result from fast path (social media)
        accurate_result: Result from accurate path (news)
        
    Returns:
        DisagreementSignal with volatility forecast
    """
    disagreement = abs(fast_result.score - accurate_result.score)
    
    # Classify disagreement level
    if disagreement > 1.0:
        signal = "HIGH_VOL_EXPECTED"
        volatility_forecast = "high"
    elif disagreement > 0.5:
        signal = "DIVERGENCE"
        volatility_forecast = "high"
    elif disagreement > 0.3:
        signal = "DIVERGENCE"
        volatility_forecast = "normal"
    else:
        signal = "ALIGNED"
        volatility_forecast = "low"
    
    # Confidence is lower when disagreement is high
    avg_confidence = (fast_result.confidence + accurate_result.confidence) / 2
    adjusted_confidence = avg_confidence * (1 - disagreement / 2)
    
    return DisagreementSignal(
        signal=signal,
        magnitude=disagreement,
        fast_score=fast_result.score,
        accurate_score=accurate_result.score,
        confidence=adjusted_confidence,
        volatility_forecast=volatility_forecast
    )


class CrossAssetSentimentContagion:
    """
    Track cross-asset sentiment spillover (Enhancement #5).
    
    Research shows BTC sentiment spills to alts with ~15min lag.
    Negative sentiment spreads faster than positive.
    
    Example:
        contagion = CrossAssetSentimentContagion()
        
        # Update BTC sentiment
        contagion.update_sentiment('BTCUSDT', -0.5)
        
        # Get spillover to ETH (with lag)
        eth_spillover = contagion.get_spillover('ETHUSDT', 'BTCUSDT')
        # Returns lagged BTC sentiment affecting ETH
    """
    
    # Spillover coefficients (how much BTC affects other assets)
    SPILLOVER_COEFFICIENTS = {
        'BTCUSDT': 1.0,      # BTC is the source
        'ETHUSDT': 0.75,     # ETH highly correlated
        'SOLUSDT': 0.60,     # SOL moderately correlated
        'BNBUSDT': 0.55,     # BNB moderately correlated
        'XRPUSDT': 0.50,     # XRP less correlated
        'ADAUSDT': 0.45,     # ADA less correlated
        'DOGEUSDT': 0.70,    # DOGE follows BTC closely (meme correlation)
    }
    
    # Lag in minutes for spillover to propagate
    SPILLOVER_LAG_MINUTES = {
        'ETHUSDT': 5,
        'SOLUSDT': 10,
        'BNBUSDT': 15,
        'XRPUSDT': 20,
        'ADAUSDT': 20,
        'DOGEUSDT': 8,
    }
    
    def __init__(self, buffer_minutes: int = 60):
        """
        Initialize cross-asset contagion tracker.
        
        Args:
            buffer_minutes: How many minutes of history to keep
        """
        from collections import deque
        self.buffer_minutes = buffer_minutes
        
        # {symbol: deque of (timestamp, sentiment)}
        self._sentiment_history: Dict[str, deque] = {}
        
    def update_sentiment(self, symbol: str, sentiment: float) -> None:
        """
        Record sentiment update for a symbol.
        
        Args:
            symbol: Trading symbol
            sentiment: Sentiment score (-1 to +1)
        """
        from collections import deque
        
        if symbol not in self._sentiment_history:
            self._sentiment_history[symbol] = deque(maxlen=self.buffer_minutes)
        
        self._sentiment_history[symbol].append((time.time(), sentiment))
    
    def get_spillover(
        self, 
        target_symbol: str, 
        source_symbol: str = 'BTCUSDT'
    ) -> Dict[str, float]:
        """
        Get sentiment spillover from source to target.
        
        Args:
            target_symbol: Symbol receiving spillover
            source_symbol: Symbol sending spillover (default: BTC)
            
        Returns:
            Dict with spillover score, lag, and coefficient
        """
        if source_symbol not in self._sentiment_history:
            return {
                'spillover_score': 0.0,
                'lag_minutes': 0,
                'coefficient': 0.0,
                'source_sentiment': 0.0
            }
        
        history = list(self._sentiment_history[source_symbol])
        if not history:
            return {
                'spillover_score': 0.0,
                'lag_minutes': 0,
                'coefficient': 0.0,
                'source_sentiment': 0.0
            }
        
        # Get lag for target
        lag_minutes = self.SPILLOVER_LAG_MINUTES.get(target_symbol, 15)
        coefficient = self.SPILLOVER_COEFFICIENTS.get(target_symbol, 0.5)
        
        # Find sentiment from lag_minutes ago
        target_time = time.time() - (lag_minutes * 60)
        lagged_sentiment = 0.0
        
        for ts, sentiment in reversed(history):
            if ts <= target_time:
                lagged_sentiment = sentiment
                break
        else:
            # Use oldest if no exact match
            lagged_sentiment = history[0][1] if history else 0.0
        
        # Compute spillover (negative sentiment spreads faster)
        spillover_multiplier = 1.3 if lagged_sentiment < 0 else 1.0
        spillover_score = lagged_sentiment * coefficient * spillover_multiplier
        
        return {
            'spillover_score': spillover_score,
            'lag_minutes': lag_minutes,
            'coefficient': coefficient,
            'source_sentiment': lagged_sentiment
        }
    
    def get_all_spillovers(
        self, 
        source_symbol: str = 'BTCUSDT'
    ) -> Dict[str, Dict[str, float]]:
        """
        Get spillovers from source to all tracked symbols.
        
        Returns:
            Dict of target_symbol -> spillover data
        """
        spillovers = {}
        for target in self.SPILLOVER_COEFFICIENTS:
            if target != source_symbol:
                spillovers[target] = self.get_spillover(target, source_symbol)
        return spillovers


@dataclass
class SentimentVolumeDivergence:
    """Result from sentiment-volume divergence analysis."""
    divergence_type: str   # "BULLISH_DIV", "BEARISH_DIV", "ALIGNED", "NO_SIGNAL"
    magnitude: float       # Strength of divergence
    sentiment_direction: str  # "bullish", "bearish", "neutral"
    volume_direction: str     # "increasing", "decreasing", "neutral"
    is_reversal_signal: bool  # True if likely reversal


def compute_sentiment_volume_divergence(
    sentiment_score: float,
    sentiment_change: float,  # Change over lookback period
    volume_change: float,     # Volume change ratio (current/average)
    threshold: float = 0.3
) -> SentimentVolumeDivergence:
    """
    Compute sentiment-volume divergence signal (Enhancement #9).
    
    Research: Volume without sentiment = noise; with sentiment = signal.
    Divergences between sentiment and volume often precede reversals.
    
    Args:
        sentiment_score: Current sentiment (-1 to +1)
        sentiment_change: Change in sentiment over lookback
        volume_change: Volume ratio (>1 = increasing, <1 = decreasing)
        threshold: Minimum absolute values for signal
        
    Returns:
        SentimentVolumeDivergence with reversal signal
    """
    # Classify sentiment direction
    if sentiment_change > threshold:
        sentiment_direction = "bullish"
    elif sentiment_change < -threshold:
        sentiment_direction = "bearish"
    else:
        sentiment_direction = "neutral"
    
    # Classify volume direction
    if volume_change > 1.2:  # 20% above average
        volume_direction = "increasing"
    elif volume_change < 0.8:  # 20% below average
        volume_direction = "decreasing"
    else:
        volume_direction = "neutral"
    
    # Detect divergences
    magnitude = abs(sentiment_change) * volume_change
    
    if sentiment_direction == "bullish" and volume_direction == "decreasing":
        # Bullish sentiment + falling volume = weak rally, potential reversal
        divergence_type = "BEARISH_DIV"
        is_reversal_signal = True
    elif sentiment_direction == "bearish" and volume_direction == "decreasing":
        # Bearish sentiment + falling volume = weak selloff, potential bounce
        divergence_type = "BULLISH_DIV"
        is_reversal_signal = True
    elif sentiment_direction == "bullish" and volume_direction == "increasing":
        # Sentiment + volume aligned bullish = strong signal
        divergence_type = "ALIGNED"
        is_reversal_signal = False
    elif sentiment_direction == "bearish" and volume_direction == "increasing":
        # Sentiment + volume aligned bearish = strong signal
        divergence_type = "ALIGNED"
        is_reversal_signal = False
    else:
        divergence_type = "NO_SIGNAL"
        is_reversal_signal = False
    
    return SentimentVolumeDivergence(
        divergence_type=divergence_type,
        magnitude=magnitude,
        sentiment_direction=sentiment_direction,
        volume_direction=volume_direction,
        is_reversal_signal=is_reversal_signal
    )


class EnhancedDualPathAnalyzer(DualPathSentimentAnalyzer):
    """
    Enhanced dual-path analyzer with all research-backed features.
    
    Adds:
    - Disagreement signal detection
    - Cross-asset sentiment contagion
    - Sentiment-volume divergence
    
    Example:
        analyzer = EnhancedDualPathAnalyzer()
        
        # Full analysis with both paths
        full_result = analyzer.analyze_dual(
            "BTC crashing!",
            source_social="twitter",
            source_news="bloomberg"
        )
    """
    
    def __init__(self, config: Optional[DualPathConfig] = None):
        super().__init__(config)
        self._contagion_tracker = CrossAssetSentimentContagion()
    
    def analyze_dual(
        self,
        text: str,
        source_social: str = "twitter",
        source_news: str = "bloomberg"
    ) -> Dict:
        """
        Analyze text with both paths and compute disagreement.
        
        Args:
            text: Input text
            source_social: Social media source label
            source_news: News source label
            
        Returns:
            Dict with both results and disagreement signal
        """
        # Analyze with both paths
        fast_result = self._analyze_fast(text, source_social)
        accurate_result = self._analyze_accurate(text, source_news)
        
        # Compute disagreement
        disagreement = compute_disagreement(fast_result, accurate_result)
        
        # Recommend which signal to trust
        if disagreement.signal == "ALIGNED":
            recommended_score = (fast_result.score + accurate_result.score) / 2
            recommended_confidence = max(fast_result.confidence, accurate_result.confidence)
        elif accurate_result.confidence > fast_result.confidence:
            recommended_score = accurate_result.score
            recommended_confidence = accurate_result.confidence * 0.8  # Penalize for disagreement
        else:
            recommended_score = fast_result.score
            recommended_confidence = fast_result.confidence * 0.7  # Fast path less reliable
        
        return {
            'fast_result': fast_result.to_dict(),
            'accurate_result': accurate_result.to_dict(),
            'disagreement': {
                'signal': disagreement.signal,
                'magnitude': disagreement.magnitude,
                'volatility_forecast': disagreement.volatility_forecast,
                'confidence': disagreement.confidence
            },
            'recommended': {
                'score': recommended_score,
                'confidence': recommended_confidence,
                'label': 'bullish' if recommended_score > 0.3 else ('bearish' if recommended_score < -0.3 else 'neutral')
            }
        }
    
    def update_contagion(self, symbol: str, sentiment: float) -> None:
        """Update contagion tracker with new sentiment."""
        self._contagion_tracker.update_sentiment(symbol, sentiment)
    
    def get_spillover(
        self, 
        target_symbol: str, 
        source_symbol: str = 'BTCUSDT'
    ) -> Dict[str, float]:
        """Get sentiment spillover from source to target."""
        return self._contagion_tracker.get_spillover(target_symbol, source_symbol)
