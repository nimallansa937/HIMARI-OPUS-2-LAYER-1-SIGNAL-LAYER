"""
HIMARI OHLCV Data Validation Module

Three-level validation framework:
- Level 1: Data Quality Checks (5 minutes) - Pass/Fail on raw data
- Level 2: Statistical Validity (30-45 minutes) - Statistical sense checks
- Level 3: Trading Signal Validation (4-8 hours) - Backtest proof (optional)

Usage:
    from validation import OHLCVValidator
    
    validator = OHLCVValidator()
    
    # Quick check (Level 1)
    is_valid, issues = validator.validate_level1(dataframe)
    
    # Full statistical check (Level 1 + 2)
    report = validator.validate_full(dataframe)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: int
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level,
            'passed': self.passed,
            'issues': self.issues,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
        }


class OHLCVValidator:
    """
    Comprehensive OHLCV data validator.
    
    Validates market data quality at three levels:
    
    Level 1 (Data Quality): 5 minutes
        - Schema correctness
        - No NaN/null values
        - Logical constraints (High >= Close >= Low)
        - No duplicates
        - Timestamps monotonic
        
    Level 2 (Statistical Validity): 30-45 minutes
        - Return distribution analysis
        - Autocorrelation checks
        - ARCH effects (volatility clustering)
        - Stationarity tests
        - Gap detection
        
    Level 3 (Trading Validation): 4-8 hours (optional)
        - Minimal signal generation
        - Backtest metrics
        - Sharpe/Win-Rate validation
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    PRICE_COLUMNS = ['open', 'high', 'low', 'close']
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    # =========================================================================
    # LEVEL 1: DATA QUALITY CHECKS (5 minutes)
    # =========================================================================
    
    def validate_level1(self, df: pd.DataFrame) -> ValidationResult:
        """
        Level 1: Basic data quality checks.
        
        Checks:
        1. Required columns present
        2. No NaN values
        3. Logical constraints (H >= C >= L, H >= L)
        4. No duplicate timestamps
        5. Timestamps monotonic (in order)
        6. Volume non-negative
        7. Prices positive
        8. No extreme outliers (>1000% single-bar change)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with pass/fail and issues
        """
        issues = []
        warnings = []
        metrics = {}
        
        logger.info("Running Level 1 validation (Data Quality)...")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 1: Required columns present
        # ─────────────────────────────────────────────────────────────────────
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(level=1, passed=False, issues=issues)
        
        metrics['row_count'] = len(df)
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 2: No NaN values
        # ─────────────────────────────────────────────────────────────────────
        nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            issues.append(f"NaN values found: {dict(nan_counts[nan_counts > 0])}")
            metrics['nan_count'] = int(total_nans)
        else:
            metrics['nan_count'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 3: Logical constraints (H >= C, C >= L, H >= L)
        # ─────────────────────────────────────────────────────────────────────
        high_close_violations = (df['high'] < df['close']).sum()
        close_low_violations = (df['close'] < df['low']).sum()
        high_low_violations = (df['high'] < df['low']).sum()
        
        if high_close_violations > 0:
            issues.append(f"High < Close violations: {high_close_violations} rows")
        if close_low_violations > 0:
            issues.append(f"Close < Low violations: {close_low_violations} rows")
        if high_low_violations > 0:
            issues.append(f"High < Low violations: {high_low_violations} rows")
        
        metrics['hlc_violations'] = int(high_close_violations + close_low_violations + high_low_violations)
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 4: No duplicate timestamps
        # ─────────────────────────────────────────────────────────────────────
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
            metrics['duplicate_timestamps'] = int(duplicates)
        else:
            metrics['duplicate_timestamps'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 5: Timestamps monotonic (in order)
        # ─────────────────────────────────────────────────────────────────────
        if len(df) > 1:
            # Handle both numeric timestamps and datetime
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                not_monotonic = (df['timestamp'].diff().iloc[1:] <= 0).sum()
            else:
                df_sorted = df.copy()
                df_sorted['ts_numeric'] = pd.to_datetime(df['timestamp']).astype(np.int64)
                not_monotonic = (df_sorted['ts_numeric'].diff().iloc[1:] <= 0).sum()
            
            if not_monotonic > 0:
                issues.append(f"Out-of-order timestamps: {not_monotonic}")
                metrics['out_of_order'] = int(not_monotonic)
            else:
                metrics['out_of_order'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 6: Volume non-negative
        # ─────────────────────────────────────────────────────────────────────
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"Negative volume values: {negative_volume}")
            metrics['negative_volumes'] = int(negative_volume)
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 7: Prices positive
        # ─────────────────────────────────────────────────────────────────────
        for col in self.PRICE_COLUMNS:
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                issues.append(f"Non-positive {col} values: {non_positive}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 8: No extreme outliers (>1000% single-bar change)
        # ─────────────────────────────────────────────────────────────────────
        if len(df) > 1:
            price_change = df['close'].pct_change().abs()
            extreme_moves = (price_change > 10.0).sum()  # >1000% change
            if extreme_moves > 0:
                warnings.append(f"Extreme price changes (>1000%): {extreme_moves} occurrences")
                metrics['extreme_moves'] = int(extreme_moves)
            else:
                metrics['extreme_moves'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────────────
        passed = len(issues) == 0
        
        result = ValidationResult(
            level=1,
            passed=passed,
            issues=issues,
            warnings=warnings,
            metrics=metrics,
        )
        
        self.results.append(result)
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"Level 1 validation: {status} ({len(issues)} issues, {len(warnings)} warnings)")
        
        return result
    
    # =========================================================================
    # LEVEL 2: STATISTICAL VALIDITY (30-45 minutes)
    # =========================================================================
    
    def validate_level2(self, df: pd.DataFrame) -> ValidationResult:
        """
        Level 2: Statistical validity checks.
        
        Checks:
        1. Return distribution (skew, kurtosis)
        2. Autocorrelation (market memory)
        3. Stationarity (ADF test on returns)
        4. ARCH effects (volatility clustering)
        5. Gap detection (missing bars)
        6. Outlier analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with statistical metrics
        """
        from scipy import stats
        
        issues = []
        warnings = []
        metrics = {}
        
        logger.info("Running Level 2 validation (Statistical Validity)...")
        
        # Ensure we have enough data
        if len(df) < 100:
            issues.append(f"Insufficient data: {len(df)} rows (need 100+)")
            return ValidationResult(level=2, passed=False, issues=issues)
        
        # Calculate log returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        metrics['return_count'] = len(returns)
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 1: Return distribution (skew, kurtosis)
        # ─────────────────────────────────────────────────────────────────────
        skew = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns)
        
        metrics['skewness'] = float(skew)
        metrics['kurtosis'] = float(kurtosis_val)
        
        if abs(skew) > 5:
            warnings.append(f"Extreme skewness ({skew:.2f}) - may indicate data errors")
        
        if kurtosis_val > 20:
            warnings.append(f"Extreme kurtosis ({kurtosis_val:.2f}) - heavy tails detected")
        elif kurtosis_val < 0:
            warnings.append(f"Negative excess kurtosis ({kurtosis_val:.2f}) - lighter tails than normal (unusual for crypto)")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 2: Autocorrelation (market should have some memory)
        # ─────────────────────────────────────────────────────────────────────
        autocorr_lag1 = returns.autocorr(lag=1)
        autocorr_lag5 = returns.autocorr(lag=5) if len(returns) > 5 else 0
        
        metrics['autocorr_lag1'] = float(autocorr_lag1) if not np.isnan(autocorr_lag1) else 0
        metrics['autocorr_lag5'] = float(autocorr_lag5) if not np.isnan(autocorr_lag5) else 0
        
        if abs(autocorr_lag1) < 0.01 and len(returns) > 1000:
            warnings.append(f"Very low autocorrelation (lag-1 = {autocorr_lag1:.4f}) - data may be artificial")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 3: Stationarity (ADF test - returns should be stationary)
        # ─────────────────────────────────────────────────────────────────────
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns, maxlag=10, autolag='AIC')
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
            
            metrics['adf_statistic'] = float(adf_stat)
            metrics['adf_pvalue'] = float(adf_pvalue)
            
            if adf_pvalue > 0.05:
                warnings.append(f"Returns may not be stationary (ADF p={adf_pvalue:.4f})")
        except ImportError:
            warnings.append("statsmodels not installed - skipping ADF test")
        except Exception as e:
            warnings.append(f"ADF test failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 4: ARCH effects (financial data should cluster volatility)
        # ─────────────────────────────────────────────────────────────────────
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_lm, arch_pvalue, _, _ = het_arch(returns.values, nlags=10)
            
            metrics['arch_lm_stat'] = float(arch_lm)
            metrics['arch_pvalue'] = float(arch_pvalue)
            
            if arch_pvalue > 0.10:
                warnings.append(f"No ARCH effects detected (p={arch_pvalue:.4f}) - volatility may not cluster (unusual)")
        except ImportError:
            warnings.append("statsmodels not installed - skipping ARCH test")
        except Exception as e:
            warnings.append(f"ARCH test failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 5: Gap detection (missing bars)
        # ─────────────────────────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            time_diffs = df['timestamp'].diff().dropna()
        else:
            time_diffs = pd.to_datetime(df['timestamp']).diff().dt.total_seconds().dropna()
        
        if len(time_diffs) > 0:
            expected_freq = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            
            # Allow 10% tolerance for expected frequency
            gap_mask = (time_diffs > expected_freq * 1.1)
            gap_count = gap_mask.sum()
            gap_ratio = gap_count / len(time_diffs)
            
            metrics['expected_bar_interval'] = float(expected_freq)
            metrics['gap_count'] = int(gap_count)
            metrics['gap_ratio'] = float(gap_ratio)
            
            if gap_ratio > 0.01:  # >1% missing
                warnings.append(f"Data gaps: {gap_ratio*100:.2f}% of bars missing ({gap_count} gaps)")
            if gap_ratio > 0.05:  # >5% missing is a problem
                issues.append(f"Excessive data gaps: {gap_ratio*100:.2f}% missing")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check 6: Basic return statistics
        # ─────────────────────────────────────────────────────────────────────
        metrics['mean_return'] = float(returns.mean())
        metrics['std_return'] = float(returns.std())
        metrics['min_return'] = float(returns.min())
        metrics['max_return'] = float(returns.max())
        
        # Sharpe approximation (annualized, assuming minute data)
        if metrics['std_return'] > 0:
            # Assume minute data, 525600 minutes/year
            annualized_return = metrics['mean_return'] * 525600
            annualized_vol = metrics['std_return'] * np.sqrt(525600)
            metrics['approx_sharpe'] = float(annualized_return / annualized_vol)
        
        # ─────────────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────────────
        passed = len(issues) == 0
        
        result = ValidationResult(
            level=2,
            passed=passed,
            issues=issues,
            warnings=warnings,
            metrics=metrics,
        )
        
        self.results.append(result)
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"Level 2 validation: {status} ({len(issues)} issues, {len(warnings)} warnings)")
        
        return result
    
    # =========================================================================
    # FULL VALIDATION (Level 1 + 2)
    # =========================================================================
    
    def validate_full(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete validation (Level 1 + Level 2).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Complete validation report
        """
        logger.info("=" * 60)
        logger.info("HIMARI OHLCV Data Validation")
        logger.info("=" * 60)
        
        # Level 1: Data Quality
        level1 = self.validate_level1(df)
        
        # Only proceed to Level 2 if Level 1 passed
        if level1.passed:
            level2 = self.validate_level2(df)
        else:
            level2 = ValidationResult(
                level=2, 
                passed=False, 
                issues=["Skipped - Level 1 failed"]
            )
        
        # Generate report
        overall_passed = level1.passed and level2.passed
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_passed': overall_passed,
            'row_count': len(df),
            'level1': level1.to_dict(),
            'level2': level2.to_dict(),
            'summary': {
                'total_issues': len(level1.issues) + len(level2.issues),
                'total_warnings': len(level1.warnings) + len(level2.warnings),
            }
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Level 1 (Data Quality): {'✓ PASS' if level1.passed else '✗ FAIL'}")
        logger.info(f"Level 2 (Statistical):  {'✓ PASS' if level2.passed else '✗ FAIL'}")
        logger.info(f"Overall: {'✓ DATA VALIDATED' if overall_passed else '✗ DATA REJECTED'}")
        logger.info("=" * 60)
        
        return report
    
    def print_report(self, report: Dict) -> None:
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("HIMARI OHLCV VALIDATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Rows analyzed: {report['row_count']:,}")
        print()
        
        # Level 1
        l1 = report['level1']
        print("─" * 40)
        print(f"LEVEL 1 - Data Quality: {'✓ PASSED' if l1['passed'] else '✗ FAILED'}")
        print("─" * 40)
        if l1['issues']:
            print("Issues:")
            for issue in l1['issues']:
                print(f"  ✗ {issue}")
        if l1['warnings']:
            print("Warnings:")
            for warning in l1['warnings']:
                print(f"  ⚠ {warning}")
        if l1['metrics']:
            print("Metrics:")
            for k, v in l1['metrics'].items():
                print(f"  • {k}: {v}")
        print()
        
        # Level 2
        l2 = report['level2']
        print("─" * 40)
        print(f"LEVEL 2 - Statistical: {'✓ PASSED' if l2['passed'] else '✗ FAILED'}")
        print("─" * 40)
        if l2['issues']:
            print("Issues:")
            for issue in l2['issues']:
                print(f"  ✗ {issue}")
        if l2['warnings']:
            print("Warnings:")
            for warning in l2['warnings']:
                print(f"  ⚠ {warning}")
        if l2['metrics']:
            print("Key Metrics:")
            for k, v in l2['metrics'].items():
                if isinstance(v, float):
                    print(f"  • {k}: {v:.6f}")
                else:
                    print(f"  • {k}: {v}")
        print()
        
        # Overall
        print("=" * 60)
        status = "✓ DATA VALIDATED" if report['overall_passed'] else "✗ DATA REJECTED"
        print(f"OVERALL: {status}")
        print("=" * 60)


# =============================================================================
# Quick validation function
# =============================================================================

def validate_ohlcv(df: pd.DataFrame, print_report: bool = True) -> Dict:
    """
    Quick function to validate OHLCV data.
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        print_report: Whether to print formatted report
        
    Returns:
        Validation report dict
    """
    validator = OHLCVValidator()
    report = validator.validate_full(df)
    
    if print_report:
        validator.print_report(report)
    
    return report


# =============================================================================
# Test with sample data
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Generate sample OHLCV data for testing
    np.random.seed(42)
    n_bars = 1000
    
    # Simulate realistic price movement
    initial_price = 100.0
    returns = np.random.normal(0, 0.002, n_bars)  # 0.2% std per bar
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1min'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
        'close': prices * (1 + np.random.normal(0, 0.0005, n_bars)),
        'volume': np.random.uniform(100, 1000, n_bars),
    })
    
    # Run validation
    print("Testing OHLCV Validator with synthetic data...")
    print()
    
    report = validate_ohlcv(df, print_report=True)
