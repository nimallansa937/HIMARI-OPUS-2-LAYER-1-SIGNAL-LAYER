"""
HIMARI Sentiment Staging Deployment
====================================

Deploys Phase 1 multi-model sentiment to staging environment.
Replaces baseline dual-path with CryptoBERT + ModernFinBERT.

Features:
- Hot-swappable model configuration
- Production monitoring with Prometheus metrics
- Agreement rate tracking (target ≥70%)
- Automatic fallback on errors

Usage:
    # Import and use in signal_processor.py
    from deployment.staging_sentiment import StagingSentimentService
    
    service = StagingSentimentService()
    result = service.analyze("BTC mooning!", source="twitter")
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class DeploymentPhase(Enum):
    """Current deployment phase."""
    BASELINE = "baseline"    # Original dual-path
    PHASE_1 = "phase_1"      # CryptoBERT + ModernFinBERT
    PHASE_2 = "phase_2"      # + FinTwitBERT + DeBERTa ensemble
    PHASE_3 = "phase_3"      # + DistilRoBERTa fallback


@dataclass
class StagingConfig:
    """Staging deployment configuration."""
    
    # Current deployment phase
    phase: DeploymentPhase = DeploymentPhase.PHASE_1
    
    # Feature flags
    enable_baseline_shadow: bool = True  # Run baseline in parallel for comparison
    enable_metrics: bool = True
    enable_fallback: bool = True
    
    # Monitoring thresholds
    min_agreement_rate: float = 0.70  # Alert if below
    max_latency_ms: float = 50.0      # Alert if above
    error_rate_threshold: float = 0.05  # 5% error rate triggers fallback
    
    # Metrics window
    metrics_window_size: int = 1000


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # Counters
    SENTIMENT_REQUESTS = Counter(
        'himari_sentiment_requests_total',
        'Total sentiment analysis requests',
        ['model', 'source', 'label']
    )
    
    SENTIMENT_ERRORS = Counter(
        'himari_sentiment_errors_total',
        'Total sentiment analysis errors',
        ['model', 'error_type']
    )
    
    # Histograms
    SENTIMENT_LATENCY = Histogram(
        'himari_sentiment_latency_seconds',
        'Sentiment analysis latency',
        ['model'],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    
    SENTIMENT_CONFIDENCE = Histogram(
        'himari_sentiment_confidence',
        'Sentiment confidence distribution',
        ['model'],
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )
    
    # Gauges
    AGREEMENT_RATE = Gauge(
        'himari_sentiment_agreement_rate',
        'Agreement rate between models'
    )
    
    CURRENT_PHASE = Gauge(
        'himari_sentiment_deployment_phase',
        'Current deployment phase (1=Phase1, 2=Phase2, etc)'
    )


# =============================================================================
# STAGING SENTIMENT SERVICE
# =============================================================================

class StagingSentimentService:
    """
    Production-ready sentiment service with hot-swappable models.
    
    Wraps multi-model ensemble with:
    - Prometheus metrics
    - Agreement rate tracking
    - Automatic fallback
    - Shadow mode comparison
    
    Example:
        service = StagingSentimentService(phase=DeploymentPhase.PHASE_1)
        
        # Analyze text
        result = service.analyze("BTC mooning!", source="twitter")
        
        # Check health
        health = service.health_check()
        print(f"Agreement rate: {health['agreement_rate']:.1%}")
    """
    
    def __init__(self, config: Optional[StagingConfig] = None):
        self.config = config or StagingConfig()
        
        # Models
        self._primary_model = None
        self._primary_model_name = "Unknown"
        self._baseline_model = None
        self._fallback_active = False
        
        # Metrics tracking
        self._agreement_window: deque = deque(maxlen=self.config.metrics_window_size)
        self._error_window: deque = deque(maxlen=self.config.metrics_window_size)
        self._latency_window: deque = deque(maxlen=self.config.metrics_window_size)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load models
        self._load_models()
        
        # Set Prometheus gauge
        if PROMETHEUS_AVAILABLE:
            phase_num = {
                DeploymentPhase.BASELINE: 0,
                DeploymentPhase.PHASE_1: 1,
                DeploymentPhase.PHASE_2: 2,
                DeploymentPhase.PHASE_3: 3,
            }
            CURRENT_PHASE.set(phase_num.get(self.config.phase, 0))
        
        logger.info(f"StagingSentimentService initialized: {self.config.phase.value}")
    
    def _load_models(self) -> None:
        """Load models based on current phase."""
        
        # Load primary model based on phase
        if self.config.phase == DeploymentPhase.BASELINE:
            self._load_baseline_as_primary()
        elif self.config.phase == DeploymentPhase.PHASE_1:
            self._load_phase1()
        elif self.config.phase == DeploymentPhase.PHASE_2:
            self._load_phase2()
        elif self.config.phase == DeploymentPhase.PHASE_3:
            self._load_phase3()
        
        # Load baseline for shadow comparison
        if self.config.enable_baseline_shadow:
            self._load_baseline_shadow()
    
    def _load_baseline_as_primary(self) -> None:
        """Load baseline dual-path as primary."""
        try:
            from primitives.dual_path_sentiment import create_dual_path_analyzer
            self._primary_model = create_dual_path_analyzer(use_fine_tuned=False)
            self._primary_model_name = "DualPath (Baseline)"
            logger.info("✓ Loaded baseline as primary")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
    
    def _load_phase1(self) -> None:
        """Load Phase 1: CryptoBERT + ModernFinBERT."""
        try:
            from primitives.multi_model_sentiment import create_phase1_analyzer
            self._primary_model = create_phase1_analyzer()
            self._primary_model_name = "MultiModel Phase 1"
            logger.info("✓ Loaded Phase 1 as primary")
        except Exception as e:
            logger.error(f"Failed to load Phase 1: {e}")
            self._load_baseline_as_primary()  # Fallback
    
    def _load_phase2(self) -> None:
        """Load Phase 2: Full ensemble with voting."""
        try:
            from primitives.multi_model_sentiment import create_phase2_analyzer
            self._primary_model = create_phase2_analyzer()
            self._primary_model_name = "MultiModel Phase 2"
            logger.info("✓ Loaded Phase 2 as primary")
        except Exception as e:
            logger.error(f"Failed to load Phase 2: {e}")
            self._load_phase1()  # Fallback
    
    def _load_phase3(self) -> None:
        """Load Phase 3: Ensemble with fallback."""
        try:
            from primitives.multi_model_sentiment import create_phase3_analyzer
            self._primary_model = create_phase3_analyzer()
            self._primary_model_name = "MultiModel Phase 3"
            logger.info("✓ Loaded Phase 3 as primary")
        except Exception as e:
            logger.error(f"Failed to load Phase 3: {e}")
            self._load_phase2()  # Fallback
    
    def _load_baseline_shadow(self) -> None:
        """Load baseline for shadow comparison."""
        try:
            from primitives.dual_path_sentiment import create_dual_path_analyzer
            self._baseline_model = create_dual_path_analyzer(use_fine_tuned=False)
            logger.info("✓ Loaded baseline for shadow comparison")
        except Exception as e:
            logger.warning(f"Shadow baseline not available: {e}")
    
    def analyze(
        self,
        text: str,
        source: str = "unknown",
        include_shadow: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze text sentiment with production monitoring.
        
        Args:
            text: Input text
            source: Source type (twitter, bloomberg, etc.)
            include_shadow: Include baseline comparison
            
        Returns:
            Dict with score, label, confidence, and metadata
        """
        start_time = time.perf_counter()
        
        result = {
            "score": 0.0,
            "label": "neutral",
            "confidence": 0.0,
            "model": self._primary_model_name,
            "latency_ms": 0.0,
            "fallback_used": self._fallback_active,
        }
        
        try:
            # Primary model prediction
            if self._primary_model:
                primary_result = self._primary_model.analyze(text, source)
                
                if primary_result:
                    result["score"] = primary_result.score
                    result["label"] = primary_result.label
                    result["confidence"] = primary_result.confidence
                    result["latency_ms"] = primary_result.latency_ms
                    
                    # Track metrics
                    self._track_success(primary_result.latency_ms)
                    
                    if PROMETHEUS_AVAILABLE:
                        SENTIMENT_REQUESTS.labels(
                            model=self._primary_model_name,
                            source=source,
                            label=primary_result.label
                        ).inc()
                        SENTIMENT_LATENCY.labels(model=self._primary_model_name).observe(
                            primary_result.latency_ms / 1000
                        )
                        SENTIMENT_CONFIDENCE.labels(model=self._primary_model_name).observe(
                            primary_result.confidence
                        )
            
            # Shadow comparison
            if include_shadow and self._baseline_model and self.config.enable_baseline_shadow:
                try:
                    baseline_result = self._baseline_model.analyze(text, source)
                    if baseline_result:
                        result["shadow_baseline"] = {
                            "score": baseline_result.score,
                            "label": baseline_result.label,
                            "confidence": baseline_result.confidence,
                        }
                        
                        # Track agreement
                        agrees = baseline_result.label == result["label"]
                        self._track_agreement(agrees)
                        
                        if PROMETHEUS_AVAILABLE:
                            AGREEMENT_RATE.set(self.get_agreement_rate())
                
                except Exception as e:
                    logger.debug(f"Shadow comparison failed: {e}")
        
        except Exception as e:
            logger.error(f"Primary model error: {e}")
            self._track_error(str(e))
            
            if PROMETHEUS_AVAILABLE:
                SENTIMENT_ERRORS.labels(
                    model=self._primary_model_name,
                    error_type=type(e).__name__
                ).inc()
            
            # Check if fallback needed
            if self.config.enable_fallback:
                self._check_fallback()
        
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def analyze_ensemble(self, text: str) -> Dict[str, Any]:
        """
        Analyze with full ensemble voting (Phase 2+).
        
        Returns detailed ensemble result with agreement and recommendation.
        """
        if self.config.phase.value < "phase_2":
            # Fall back to single model
            return self.analyze(text)
        
        try:
            ensemble_result = self._primary_model.analyze_ensemble(text)
            
            return {
                "score": ensemble_result.final_score,
                "label": ensemble_result.final_label,
                "confidence": ensemble_result.final_confidence,
                "agreement_rate": ensemble_result.agreement_rate,
                "confidence_level": ensemble_result.confidence_level,
                "position_recommendation": ensemble_result.position_recommendation,
                "predictions": [p.to_dict() for p in ensemble_result.individual_predictions],
                "total_latency_ms": ensemble_result.total_latency_ms,
            }
        
        except Exception as e:
            logger.error(f"Ensemble analysis failed: {e}")
            return self.analyze(text)
    
    def _track_success(self, latency_ms: float) -> None:
        """Track successful request."""
        with self._lock:
            self._latency_window.append(latency_ms)
            self._error_window.append(0)
    
    def _track_error(self, error: str) -> None:
        """Track error."""
        with self._lock:
            self._error_window.append(1)
    
    def _track_agreement(self, agrees: bool) -> None:
        """Track model agreement."""
        with self._lock:
            self._agreement_window.append(1 if agrees else 0)
    
    def _check_fallback(self) -> None:
        """Check if fallback should be activated."""
        with self._lock:
            if len(self._error_window) < 10:
                return
            
            error_rate = sum(self._error_window) / len(self._error_window)
            
            if error_rate > self.config.error_rate_threshold:
                if not self._fallback_active:
                    logger.warning(f"Activating fallback: error rate {error_rate:.1%}")
                    self._fallback_active = True
                    # Swap to baseline
                    if self._baseline_model:
                        self._primary_model = self._baseline_model
                        self._primary_model_name = "DualPath (Fallback)"
    
    def get_agreement_rate(self) -> float:
        """Get current agreement rate."""
        with self._lock:
            if not self._agreement_window:
                return 1.0
            return sum(self._agreement_window) / len(self._agreement_window)
    
    def get_error_rate(self) -> float:
        """Get current error rate."""
        with self._lock:
            if not self._error_window:
                return 0.0
            return sum(self._error_window) / len(self._error_window)
    
    def get_latency_p50(self) -> float:
        """Get p50 latency."""
        with self._lock:
            if not self._latency_window:
                return 0.0
            import numpy as np
            return np.percentile(list(self._latency_window), 50)
    
    def get_latency_p95(self) -> float:
        """Get p95 latency."""
        with self._lock:
            if not self._latency_window:
                return 0.0
            import numpy as np
            return np.percentile(list(self._latency_window), 95)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Get service health status.
        
        Returns:
            Dict with health metrics and status
        """
        agreement_rate = self.get_agreement_rate()
        error_rate = self.get_error_rate()
        latency_p50 = self.get_latency_p50()
        latency_p95 = self.get_latency_p95()
        
        # Determine health status
        issues = []
        
        if agreement_rate < self.config.min_agreement_rate:
            issues.append(f"Low agreement rate: {agreement_rate:.1%}")
        
        if error_rate > self.config.error_rate_threshold:
            issues.append(f"High error rate: {error_rate:.1%}")
        
        if latency_p95 > self.config.max_latency_ms:
            issues.append(f"High latency p95: {latency_p95:.1f}ms")
        
        status = "healthy" if not issues else "degraded"
        if self._fallback_active:
            status = "fallback"
        
        return {
            "status": status,
            "phase": self.config.phase.value,
            "model": self._primary_model_name,
            "agreement_rate": agreement_rate,
            "error_rate": error_rate,
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "fallback_active": self._fallback_active,
            "issues": issues,
            "samples_processed": len(self._latency_window),
        }
    
    def switch_phase(self, phase: DeploymentPhase) -> bool:
        """
        Hot-switch to a different deployment phase.
        
        Args:
            phase: New deployment phase
            
        Returns:
            True if switch successful
        """
        logger.info(f"Switching from {self.config.phase.value} to {phase.value}")
        
        old_phase = self.config.phase
        self.config.phase = phase
        
        try:
            self._load_models()
            
            # Clear metrics windows
            with self._lock:
                self._agreement_window.clear()
                self._error_window.clear()
                self._latency_window.clear()
                self._fallback_active = False
            
            if PROMETHEUS_AVAILABLE:
                phase_num = {
                    DeploymentPhase.BASELINE: 0,
                    DeploymentPhase.PHASE_1: 1,
                    DeploymentPhase.PHASE_2: 2,
                    DeploymentPhase.PHASE_3: 3,
                }
                CURRENT_PHASE.set(phase_num.get(phase, 0))
            
            logger.info(f"Successfully switched to {phase.value}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to switch phase: {e}")
            self.config.phase = old_phase
            self._load_models()
            return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_staging_service(phase: str = "phase_1") -> StagingSentimentService:
    """Create staging service with specified phase."""
    phase_map = {
        "baseline": DeploymentPhase.BASELINE,
        "phase_1": DeploymentPhase.PHASE_1,
        "phase_2": DeploymentPhase.PHASE_2,
        "phase_3": DeploymentPhase.PHASE_3,
    }
    
    config = StagingConfig(phase=phase_map.get(phase, DeploymentPhase.PHASE_1))
    return StagingSentimentService(config)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("STAGING SENTIMENT SERVICE TEST")
    print("=" * 60)
    
    # Create Phase 1 service
    service = create_staging_service("phase_1")
    
    # Test samples
    tests = [
        ("BTC mooning! 🚀🚀🚀", "twitter"),
        ("SEC crackdown incoming", "twitter"),
        ("Bitcoin surges past $50k", "bloomberg"),
    ]
    
    print("\n🧪 Testing sentiment analysis...\n")
    
    for text, source in tests:
        result = service.analyze(text, source)
        print(f"[{source:10}] {text[:30]}...")
        print(f"   Score: {result['score']:+.3f} | Label: {result['label']}")
        print(f"   Model: {result['model']}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        if "shadow_baseline" in result:
            shadow = result["shadow_baseline"]
            print(f"   Shadow: {shadow['label']} ({shadow['score']:+.3f})")
        print()
    
    # Health check
    health = service.health_check()
    print("=" * 60)
    print("HEALTH CHECK")
    print("=" * 60)
    print(f"Status: {health['status']}")
    print(f"Phase: {health['phase']}")
    print(f"Agreement rate: {health['agreement_rate']:.1%}")
    print(f"Latency p50: {health['latency_p50_ms']:.1f}ms")
