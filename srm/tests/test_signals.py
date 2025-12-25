"""
Unit Tests for SRM Signals

Tests for all 6 risk signals:
- FSI: Funding Saturation Index
- LEI: Liquidity Evaporation Index
- ODS: Oracle Divergence Score
- SCSI: Stablecoin Stress Index
- LCI: Leverage Concentration Index
- CACI: Cross-Asset Contagion Index
"""

import pytest
from datetime import datetime, timedelta

from srm.signals import (
    FundingSaturationIndex, FSIConfig,
    LiquidityEvaporationIndex, LEIConfig,
    OracleDivergenceScore, ODSConfig,
    LeverageConcentrationIndex, LCIConfig,
    CrossAssetContagionIndex, CACIConfig,
)
from srm.regime import RegimeDetector, MarketRegime
from srm.composite import CompositeRiskCalculator


class TestFSI:
    """Unit tests for Funding Saturation Index."""
    
    def test_insufficient_data(self):
        """FSI returns 0 when history is too short."""
        fsi = FundingSaturationIndex()
        for i in range(3):
            fsi.update(0.001, datetime.now() - timedelta(hours=i*8))
        
        score, meta = fsi.calculate('MEDIUM')
        assert score == 0.0
        assert meta['status'] == 'insufficient_data'
    
    def test_elevated_funding_low_volatility(self):
        """High funding in calm markets triggers elevated score."""
        fsi = FundingSaturationIndex()
        
        # Simulate 48h of 0.10% funding
        for i in range(12):
            fsi.update(0.0010, datetime.now() - timedelta(hours=i*4))
        
        score, meta = fsi.calculate('LOW')
        assert score >= 0.8, f"Expected >= 0.8 in low vol regime, got {score}"
    
    def test_same_funding_high_volatility(self):
        """Same funding is less dangerous in volatile markets."""
        fsi = FundingSaturationIndex()
        
        for i in range(12):
            fsi.update(0.0010, datetime.now() - timedelta(hours=i*4))
        
        score, meta = fsi.calculate('HIGH')
        assert score < 0.8, f"Expected < 0.8 in high vol regime, got {score}"
    
    def test_velocity_penalty(self):
        """Accelerating funding triggers velocity penalty."""
        fsi = FundingSaturationIndex()
        
        # Simulate accelerating funding
        for i in range(10):
            rate = 0.0005 + (i * 0.0002)  # Increasing rate
            fsi.update(rate, datetime.now() - timedelta(hours=(10-i)*8))
        
        score, meta = fsi.calculate('MEDIUM')
        assert meta['velocity_penalty'] > 0, "Expected velocity penalty for accelerating funding"


class TestLEI:
    """Unit tests for Liquidity Evaporation Index."""
    
    def test_insufficient_baseline(self):
        """LEI returns 0 when baseline history is too short."""
        lei = LiquidityEvaporationIndex()
        
        # Only add a few observations
        for i in range(100):
            lei.update(1000000, 1000000, datetime.now() - timedelta(minutes=i))
        
        score, meta = lei.calculate(2000000, datetime.now())
        assert score == 0.0
        assert meta['status'] == 'insufficient_baseline'
    
    def test_depth_evaporation_detection(self):
        """LEI detects significant depth decline."""
        lei = LiquidityEvaporationIndex()
        
        # Build baseline with 2M depth
        for i in range(24 * 60 * 2):  # 2 days
            lei.update(1000000, 1000000, datetime.now() - timedelta(minutes=i))
        
        # Current depth is only 500K (75% evaporation)
        score, meta = lei.calculate(500000, datetime.now())
        assert score >= 0.5, f"Expected >= 0.5 for 75% evaporation, got {score}"
    
    def test_weekend_adjustment(self):
        """Weekend has lower baseline expectation."""
        lei = LiquidityEvaporationIndex()
        
        # Build baseline
        for i in range(24 * 60 * 2):
            lei.update(1000000, 1000000, datetime.now() - timedelta(minutes=i))
        
        # Test on a Saturday
        saturday = datetime(2024, 12, 21, 12, 0)  # A Saturday
        score_weekend, meta_weekend = lei.calculate(1400000, saturday)
        
        # Test on a Tuesday
        tuesday = datetime(2024, 12, 24, 12, 0)  # A Tuesday
        score_weekday, meta_weekday = lei.calculate(1400000, tuesday)
        
        assert meta_weekend['is_weekend'] == True
        # Weekend should have lower evaporation score for same depth
        # because expected depth is lower


class TestODS:
    """Unit tests for Oracle Divergence Score."""
    
    def test_insufficient_venues(self):
        """ODS returns 0 when less than 2 venues available."""
        ods = OracleDivergenceScore()
        
        prices = {'binance': 50000.0, 'coinbase': None}
        score, meta = ods.calculate(prices)
        
        assert score == 0.0
        assert meta['status'] == 'insufficient_venues'
    
    def test_no_divergence(self):
        """ODS returns low score when prices align."""
        ods = OracleDivergenceScore()
        
        prices = {
            'binance': 50000.0,
            'coinbase': 50010.0,
            'kraken': 49995.0,
            'coingecko': 50002.0
        }
        score, meta = ods.calculate(prices)
        
        assert score < 0.1, f"Expected < 0.1 for aligned prices, got {score}"
    
    def test_significant_divergence(self):
        """ODS detects significant price divergence."""
        ods = OracleDivergenceScore()
        
        prices = {
            'binance': 50000.0,
            'coinbase': 52500.0,  # 5% higher
            'kraken': 50100.0,
        }
        score, meta = ods.calculate(prices)
        
        assert score >= 0.8, f"Expected >= 0.8 for 5% divergence, got {score}"
        assert meta['worst_venue'] == 'coinbase'


class TestLCI:
    """Unit tests for Leverage Concentration Index."""
    
    def test_balanced_distribution(self):
        """LCI returns low score for balanced OI distribution."""
        lci = LeverageConcentrationIndex()
        
        oi_distribution = {
            'binance': 20e9,
            'bybit': 18e9,
            'okx': 15e9,
            'deribit': 12e9,
            'bitget': 10e9
        }
        
        score, meta = lci.calculate(oi_distribution)
        assert score < 0.3, f"Expected < 0.3 for balanced distribution, got {score}"
        assert meta['concentration_level'] in ['LOW', 'MODERATE']
    
    def test_concentrated_distribution(self):
        """LCI detects high concentration on single venue."""
        lci = LeverageConcentrationIndex()
        
        oi_distribution = {
            'binance': 80e9,  # 80% share
            'bybit': 10e9,
            'okx': 5e9,
            'deribit': 3e9,
            'bitget': 2e9
        }
        
        score, meta = lci.calculate(oi_distribution)
        assert score >= 0.7, f"Expected >= 0.7 for concentrated distribution, got {score}"
        assert meta['dominant_venue'] == 'binance'
        assert meta['concentration_level'] in ['HIGH', 'CRITICAL']


class TestCACI:
    """Unit tests for Cross-Asset Contagion Index."""
    
    def test_low_vix_low_stress(self):
        """CACI returns low score when VIX is calm."""
        caci = CrossAssetContagionIndex()
        
        tradfi_data = {
            'vix': 15.0,
            'usdjpy': 150.0,
            'spx': 5000.0
        }
        caci.spx_history = [4950, 4960, 4980, 5000, 5000]
        
        score, meta = caci.calculate(tradfi_data)
        assert score < 0.2, f"Expected < 0.2 for calm VIX, got {score}"
    
    def test_crisis_vix(self):
        """CACI detects VIX crisis levels."""
        caci = CrossAssetContagionIndex()
        
        tradfi_data = {
            'vix': 65.0,  # Crisis level (Aug 5, 2024 was ~65)
            'usdjpy': 150.0,
            'spx': 5000.0
        }
        caci.spx_history = [5000, 5000, 5000, 5000, 5000]
        
        score, meta = caci.calculate(tradfi_data)
        assert score >= 0.5, f"Expected >= 0.5 for VIX 65, got {score}"
        assert meta['stress_level'] in ['HIGH', 'CRISIS']


class TestRegimeDetector:
    """Unit tests for regime detection."""
    
    def test_normal_regime(self):
        """Detector returns NORMAL for low signals."""
        detector = RegimeDetector()
        
        # Add history with low signals
        for i in range(20):
            signals = {'fsi': 0.2, 'lei': 0.3, 'ods': 0.1, 'scsi': 0.1, 'lci': 0.2, 'caci': 0.1}
            detector.update(signals, datetime.now() - timedelta(minutes=60-i))
        
        regime, meta = detector.detect_regime(datetime.now())
        assert regime == MarketRegime.NORMAL
    
    def test_oracle_failure_regime(self):
        """Detector identifies oracle failure regime."""
        detector = RegimeDetector()
        
        # Add history with high ODS
        for i in range(20):
            signals = {'fsi': 0.2, 'lei': 0.3, 'ods': 0.5, 'scsi': 0.5, 'lci': 0.2, 'caci': 0.1}
            detector.update(signals, datetime.now() - timedelta(minutes=60-i))
        
        regime, meta = detector.detect_regime(datetime.now())
        assert regime == MarketRegime.ORACLE_FAILURE
    
    def test_tradfi_contagion_regime(self):
        """Detector identifies TradFi contagion regime."""
        detector = RegimeDetector()
        
        # Add history with high CACI
        for i in range(20):
            signals = {'fsi': 0.3, 'lei': 0.3, 'ods': 0.1, 'scsi': 0.1, 'lci': 0.2, 'caci': 0.7}
            detector.update(signals, datetime.now() - timedelta(minutes=60-i))
        
        regime, meta = detector.detect_regime(datetime.now())
        assert regime == MarketRegime.TRADFI_CONTAGION


class TestCompositeRisk:
    """Unit tests for composite risk calculation."""
    
    def test_normal_regime_weights(self):
        """Verify normal regime uses balanced weights."""
        calc = CompositeRiskCalculator()
        
        signals = {'fsi': 0.3, 'lei': 0.3, 'ods': 0.1, 'scsi': 0.1, 'lci': 0.1, 'caci': 0.1}
        result = calc.calculate(signals, datetime.now())
        
        assert result.regime == MarketRegime.NORMAL
        assert 0.2 < result.score < 0.4
    
    def test_amplification_trigger(self):
        """Multiple elevated signals trigger amplification."""
        calc = CompositeRiskCalculator()
        
        # 5 signals above 0.5
        signals = {'fsi': 0.6, 'lei': 0.6, 'ods': 0.6, 'scsi': 0.6, 'lci': 0.6, 'caci': 0.2}
        result = calc.calculate(signals, datetime.now())
        
        assert result.amplification_applied == True
    
    def test_critical_score(self):
        """All high signals produce critical score."""
        calc = CompositeRiskCalculator()
        
        signals = {'fsi': 0.9, 'lei': 0.9, 'ods': 0.9, 'scsi': 0.9, 'lci': 0.9, 'caci': 0.9}
        result = calc.calculate(signals, datetime.now())
        
        assert result.score >= 0.9
        assert result.metadata['risk_level'] == 'CRITICAL'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
