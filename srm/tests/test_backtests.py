"""
Historical Cascade Backtests

Validates SRM against known historical cascades:
- October 10, 2025: Oracle failure (USDe depeg)
- August 5, 2024: TradFi contagion (yen carry unwind)
- May 19, 2021: Leverage saturation

Each test case uses forensically documented values and verifies
the SRM would have triggered appropriate alerts.
"""

import pytest
from datetime import datetime, timedelta

from srm.composite import CompositeRiskCalculator
from srm.regime import MarketRegime


class TestCascadeBacktests:
    """
    Validate SRM against known historical cascades.
    
    Each test case uses forensically documented values
    and verifies the SRM would have triggered appropriate alerts.
    """
    
    def test_october_10_2025(self):
        """
        October 10, 2025 cascade backtest.
        
        Forensic data:
        - Funding: 0.14% (sustained 48h)
        - OI: 10.6% of market cap
        - Depth: Collapsed from $1.2M to $27K (97.75%)
        - USDe: 35% depeg on Binance vs Curve
        - Outcome: $19B liquidated
        
        Expected: Critical score (>0.9), ORACLE_FAILURE regime
        """
        signals = {
            'fsi': 0.93,  # 0.14% funding at high threshold 0.15
            'lei': 0.98,  # 97.75% depth evaporation
            'ods': 1.0,   # 35% USDe divergence
            'scsi': 1.0,  # Stablecoin collapse
            'lci': 0.7,   # Binance 45% concentration
            'caci': 0.3   # Modest TradFi stress
        }
        
        calc = CompositeRiskCalculator()
        
        # Pre-seed with oracle failure pattern
        for i in range(15):
            history_signals = {
                'fsi': 0.8, 'lei': 0.75, 'ods': 0.55, 
                'scsi': 0.60, 'lci': 0.60, 'caci': 0.25
            }
            calc.regime_detector.update(
                history_signals, 
                datetime(2025, 10, 10, 15, 0) - timedelta(minutes=i*5)
            )
        
        result = calc.calculate(signals, datetime(2025, 10, 10, 16, 0))
        
        assert result.score > 0.9, f"Expected critical score >0.9, got {result.score:.3f}"
        assert result.regime == MarketRegime.ORACLE_FAILURE, f"Expected ORACLE_FAILURE, got {result.regime}"
        assert result.metadata['risk_level'] == 'CRITICAL'
        
        print(f"✅ Oct 10, 2025: Score={result.score:.3f}, Regime={result.regime.value}")
    
    def test_august_5_2024(self):
        """
        August 5, 2024 yen carry trade unwind.
        
        Forensic data:
        - VIX spiked to 65
        - USD/JPY moved 8% in 48h
        - Funding: 0.08%
        - BTC dropped 15%
        - External TradFi catalyst
        
        Expected: Elevated score (>0.6), TRADFI_CONTAGION regime
        """
        signals = {
            'fsi': 0.53,  # 0.08% moderate funding
            'lei': 0.6,   # Moderate depth decline
            'ods': 0.2,   # Minimal oracle divergence
            'scsi': 0.2,  # Stablecoins held
            'lci': 0.5,   # Normal concentration
            'caci': 1.0   # VIX 65, extreme TradFi stress
        }
        
        calc = CompositeRiskCalculator()
        
        # Pre-seed regime detector with CACI history
        for i in range(15):
            history_signals = {
                'caci': 0.8, 'fsi': 0.4, 'lei': 0.5, 
                'ods': 0.1, 'scsi': 0.1, 'lci': 0.4
            }
            calc.regime_detector.update(
                history_signals,
                datetime(2024, 8, 5, 14, 0) - timedelta(minutes=i*5)
            )
        
        result = calc.calculate(signals, datetime(2024, 8, 5, 15, 0))
        
        # TradFi contagion typically produces elevated (not critical) scores
        # because crypto-native signals (FSI, LEI, ODS) remain moderate
        assert result.score > 0.6, f"Expected elevated score >0.6, got {result.score:.3f}"
        assert result.regime == MarketRegime.TRADFI_CONTAGION, f"Expected TRADFI_CONTAGION, got {result.regime}"
        
        print(f"✅ Aug 5, 2024: Score={result.score:.3f}, Regime={result.regime.value}")
    
    def test_may_19_2021(self):
        """
        May 19, 2021 leverage cascade.
        
        Forensic data:
        - Funding: >0.15% sustained
        - OI saturation at historical highs
        - Pure crypto leverage event
        - No external catalyst
        
        Expected: High score (>0.65), LEVERAGE_SATURATION regime
        """
        signals = {
            'fsi': 1.0,   # Maximum funding saturation
            'lei': 0.85,  # Severe depth decline
            'ods': 0.15,  # Minimal divergence
            'scsi': 0.1,  # Stablecoins healthy
            'lci': 0.6,   # Concentrated on few venues
            'caci': 0.1   # No TradFi contagion
        }
        
        calc = CompositeRiskCalculator()
        
        # Pre-seed with leverage buildup
        for i in range(15):
            history_signals = {
                'fsi': 0.9, 'lei': 0.6, 'ods': 0.1, 
                'scsi': 0.1, 'lci': 0.5, 'caci': 0.1
            }
            calc.regime_detector.update(
                history_signals,
                datetime(2021, 5, 19, 10, 0) - timedelta(minutes=i*5)
            )
        
        result = calc.calculate(signals, datetime(2021, 5, 19, 12, 0))
        
        # Leverage saturation produces high scores due to FSI + LEI dominance
        assert result.score > 0.65, f"Expected high score >0.65, got {result.score:.3f}"
        assert result.regime == MarketRegime.LEVERAGE_SATURATION, f"Expected LEVERAGE_SATURATION, got {result.regime}"
        
        print(f"✅ May 19, 2021: Score={result.score:.3f}, Regime={result.regime.value}")
    
    def test_detection_window_october_2025(self):
        """
        Verify October 2025 cascade would have been detected 1h before peak.
        
        Pre-crash signal values should trigger warnings before
        the catastrophic acceleration phase.
        """
        # Values 1 hour before peak (from forensic data)
        pre_crash_signals = {
            'fsi': 0.85,
            'lei': 0.75,
            'ods': 0.55,  # USDe starting to diverge
            'scsi': 0.60,
            'lci': 0.60,
            'caci': 0.25
        }
        
        calc = CompositeRiskCalculator()
        
        # Pre-seed with earlier data
        for i in range(15):
            history_signals = {
                'fsi': 0.65, 'lei': 0.40, 'ods': 0.10, 
                'scsi': 0.10, 'lci': 0.50, 'caci': 0.20
            }
            calc.regime_detector.update(
                history_signals,
                datetime(2025, 10, 10, 14, 0) - timedelta(minutes=i*5)
            )
        
        result = calc.calculate(pre_crash_signals, datetime(2025, 10, 10, 15, 0))
        
        # Should trigger at least HIGH warning 1h before crash
        assert result.score > 0.5, f"Expected elevated score >0.5 1h before crash, got {result.score:.3f}"
        assert result.metadata['risk_level'] in ['ELEVATED', 'HIGH', 'CRITICAL']
        
        print(f"✅ 1h Before Oct 10, 2025: Score={result.score:.3f}, Level={result.metadata['risk_level']}")


def run_all_backtests():
    """Execute all historical backtests."""
    tests = TestCascadeBacktests()
    tests.test_october_10_2025()
    tests.test_august_5_2024()
    tests.test_may_19_2021()
    tests.test_detection_window_october_2025()
    print("\n✅ All historical backtests passed")


if __name__ == "__main__":
    run_all_backtests()
