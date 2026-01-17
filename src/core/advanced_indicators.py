"""
Advanced Non-Lag Technical Indicators

Implements sophisticated indicators with lower lag and better signal quality than
traditional moving averages. These indicators are designed for high-frequency
trading and require high-quality OHLCV data.

All functions are optimized with numba where possible for real-time computation.
"""

import numpy as np
from numba import jit
from typing import Optional
import pandas as pd


@jit(nopython=True)
def _jma_core(prices: np.ndarray, period: int = 14, phase: float = 0.0) -> np.ndarray:
    """
    Jurik Moving Average (JMA) - Zero-lag adaptive moving average.

    Uses a Kalman-like filter with phase control to eliminate lag while
    maintaining smoothness. Superior to EMA for trend following.

    Args:
        prices: Price array
        period: Smoothing period (default 14)
        phase: Phase parameter from -100 to +100 (0 = balanced)

    Returns:
        JMA values
    """
    n = len(prices)
    jma = np.zeros(n)

    # Simplified JMA algorithm (full implementation is proprietary)
    # This is a close approximation using adaptive smoothing
    alpha = 2.0 / (period + 1.0)
    beta = phase / 100.0  # Normalize phase

    # Initialize
    jma[0] = prices[0]
    ema1 = prices[0]
    ema2 = prices[0]

    for i in range(1, n):
        # Double exponential smoothing with phase adjustment
        ema1 = alpha * prices[i] + (1 - alpha) * ema1
        ema2 = alpha * ema1 + (1 - alpha) * ema2

        # Phase-adjusted output
        jma[i] = (1 + beta) * ema1 - beta * ema2

    return jma


def jurik_ma(prices: pd.Series, period: int = 14, phase: float = 0.0) -> pd.Series:
    """Jurik Moving Average - zero lag adaptive MA."""
    result = _jma_core(prices.values, period, phase)
    return pd.Series(result, index=prices.index)


@jit(nopython=True)
def _kama_core(prices: np.ndarray, period: int = 14, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (KAMA).

    Adapts smoothing based on market efficiency ratio. Moves faster in
    trending markets, slower in ranging markets.

    Args:
        prices: Price array
        period: Efficiency ratio period
        fast: Fast EMA period (default 2)
        slow: Slow EMA period (default 30)

    Returns:
        KAMA values
    """
    n = len(prices)
    kama = np.zeros(n)
    kama[0] = prices[0]

    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    for i in range(period, n):
        # Calculate efficiency ratio
        change = abs(prices[i] - prices[i - period])
        volatility = np.sum(np.abs(np.diff(prices[i-period:i+1])))

        if volatility > 0:
            er = change / volatility  # Efficiency ratio [0, 1]
        else:
            er = 0.0

        # Scaled smoothing constant
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # Adaptive smoothing
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])

    # Fill initial values
    kama[:period] = prices[:period]

    return kama


def kaufman_adaptive_ma(prices: pd.Series, period: int = 14) -> pd.Series:
    """Kaufman Adaptive Moving Average - efficiency-weighted."""
    result = _kama_core(prices.values, period)
    return pd.Series(result, index=prices.index)


@jit(nopython=True)
def _hma_core(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Hull Moving Average (HMA) - Low-lag weighted MA.

    Combines weighted moving averages to achieve smoothness with minimal lag.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    Args:
        prices: Price array
        period: Smoothing period

    Returns:
        HMA values
    """
    def wma(data, window):
        """Weighted moving average."""
        weights = np.arange(1, window + 1)
        result = np.zeros(len(data))
        for i in range(window - 1, len(data)):
            result[i] = np.sum(data[i-window+1:i+1] * weights) / weights.sum()
        return result

    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))

    wma_half = wma(prices, half_period)
    wma_full = wma(prices, period)

    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)

    return hma


def hull_ma(prices: pd.Series, period: int = 14) -> pd.Series:
    """Hull Moving Average - low-lag weighted MA."""
    result = _hma_core(prices.values, period)
    return pd.Series(result, index=prices.index)


def fisher_transform(prices_high: pd.Series, prices_low: pd.Series, period: int = 10) -> pd.Series:
    """
    Fisher Transform - Converts price to Gaussian distribution.

    Normalizes price to [-1, 1] then applies inverse hyperbolic tangent
    to create sharp, clear turning points.

    Args:
        prices_high: High prices
        prices_low: Low prices
        period: Lookback period for normalization

    Returns:
        Fisher transform values
    """
    hl2 = (prices_high + prices_low) / 2

    min_low = hl2.rolling(period).min()
    max_high = hl2.rolling(period).max()

    # Normalize to [-1, 1]
    value = 2 * ((hl2 - min_low) / (max_high - min_low + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)  # Prevent overflow in arctanh

    # Fisher transform (inverse hyperbolic tangent)
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))

    # Smooth the output
    fisher_smooth = fisher.ewm(span=3).mean()

    return fisher_smooth


def keltner_position(close: pd.Series, high: pd.Series, low: pd.Series,
                     period: int = 20, atr_mult: float = 2.0) -> pd.Series:
    """
    Position within Keltner Channels (z-score).

    Keltner Channels use ATR-based bands around EMA. This returns
    the z-score position of price within the channels.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        period: EMA period
        atr_mult: ATR multiplier for bands

    Returns:
        Z-score position (-3 to +3, where 0 = at EMA)
    """
    # Calculate EMA basis
    basis = close.ewm(span=period, adjust=False).mean()

    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Channel bands
    upper = basis + atr_mult * atr
    lower = basis - atr_mult * atr

    # Z-score position (0 = at basis, +atr_mult = at upper, -atr_mult = at lower)
    channel_width = upper - lower
    position = (close - basis) / (channel_width / (2 * atr_mult) + 1e-10)

    return position.clip(-3, 3)


def garman_klass_volatility(open_: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series,
                            window: int = 14) -> pd.Series:
    """
    Garman-Klass Volatility Estimator.

    Uses OHLC data to estimate volatility more accurately than close-to-close.
    About 5x more efficient than standard deviation of returns.

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window for averaging

    Returns:
        Annualized volatility estimate
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    # Garman-Klass formula
    gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2

    # Rolling average and annualization (assuming 24/7 trading)
    gk_vol = np.sqrt(gk.rolling(window).mean() * 365)

    return gk_vol


def vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 50) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    Estimates the probability of informed trading based on order flow imbalance.
    Higher VPIN indicates more informed trading (potential toxic flow).

    Args:
        volume: Total volume
        buy_volume: Buy-side volume
        window: Number of volume buckets to average

    Returns:
        VPIN values [0, 1]
    """
    # Calculate sell volume
    sell_volume = volume - buy_volume

    # Order imbalance
    imbalance = abs(buy_volume - sell_volume)

    # VPIN = average imbalance / average volume
    avg_imbalance = imbalance.rolling(window).sum()
    avg_volume = volume.rolling(window).sum()

    vpin_values = avg_imbalance / (avg_volume + 1e-10)

    return vpin_values.clip(0, 1)


def vwap_distance(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Distance from VWAP as percentage.

    Measures how far price has deviated from volume-weighted average price.
    Useful for mean reversion and microstructure analysis.

    Args:
        close: Close prices
        volume: Volume
        period: Lookback period for VWAP

    Returns:
        Distance as percentage (-0.1 to +0.1 typical)
    """
    typical_price = close
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()

    distance = (close - vwap) / vwap

    return distance.clip(-0.1, 0.1)


def instantaneous_trend(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Instantaneous Trendline (Ehlers).

    Uses Hilbert Transform to extract the instantaneous trend component
    from price action. Based on DSP principles.

    Args:
        close: Close prices
        period: Dominant cycle period estimate

    Returns:
        Instantaneous trend rate of change
    """
    # Simplified version using phase accumulation
    # Full Hilbert Transform implementation requires complex DSP

    # Smooth price
    smooth = close.ewm(span=period//2, adjust=False).mean()

    # Calculate instantaneous phase
    detrender = smooth - smooth.shift(period//2)

    # Normalize
    trend = detrender / (smooth + 1e-10)

    return trend.clip(-0.2, 0.2)


def dominant_cycle_period(close: pd.Series, min_period: int = 10, max_period: int = 50) -> pd.Series:
    """
    Dominant Cycle Period Detector (MESA Adaptive).

    Estimates the current dominant market cycle period using autocorrelation.
    Helps adapt indicator periods to market conditions.

    Args:
        close: Close prices
        min_period: Minimum cycle period to search
        max_period: Maximum cycle period to search

    Returns:
        Dominant cycle period in bars
    """
    def autocorr(x, lag):
        """Autocorrelation at given lag."""
        if len(x) < lag + 1:
            return 0.0
        c0 = np.var(x)
        c1 = np.corrcoef(x[:-lag], x[lag:])[0, 1] * c0
        return c1 / c0 if c0 > 0 else 0.0

    periods = []
    window = max_period * 2

    for i in range(window, len(close)):
        segment = close.iloc[i-window:i].values

        # Find period with highest autocorrelation
        max_corr = -1
        best_period = min_period

        for period in range(min_period, max_period + 1):
            corr = autocorr(segment, period)
            if corr > max_corr:
                max_corr = corr
                best_period = period

        periods.append(best_period)

    # Pad initial values
    result = pd.Series([min_period] * window + periods, index=close.index)

    return result


# ============== Helper function to compute all advanced indicators ==============

def compute_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 advanced indicators from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, buy_volume

    Returns:
        DataFrame with 10 new columns (indices 60-69 in feature vector)
    """
    result = df.copy()

    # Adaptive Moving Averages
    result['jma_14'] = jurik_ma(df['close'], period=14)
    result['kama_14'] = kaufman_adaptive_ma(df['close'], period=14)
    result['hma_14'] = hull_ma(df['close'], period=14)

    # Advanced Mean Reversion
    result['fisher_transform'] = fisher_transform(df['high'], df['low'], period=10)
    result['keltner_position'] = keltner_position(df['close'], df['high'], df['low'], period=20)

    # Advanced Volatility
    result['garman_klass_vol'] = garman_klass_volatility(df['open'], df['high'], df['low'], df['close'], window=14)

    # Market Microstructure
    result['vpin'] = vpin(df['volume'], df.get('buy_volume', df['volume'] * 0.5), window=50)
    result['vwap_distance'] = vwap_distance(df['close'], df['volume'], period=20)

    # Ehlers Cycle Analysis
    result['instantaneous_trend'] = instantaneous_trend(df['close'], period=20)
    result['dominant_cycle_period'] = dominant_cycle_period(df['close'], min_period=10, max_period=50)

    return result
