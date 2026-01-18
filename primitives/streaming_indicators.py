"""
Advanced Low-Lag Streaming Indicators

Replaced traditional lagging indicators with Ehlers/Jurik DSP-based alternatives:
- EMA/SMA → Jurik Moving Average (JMA), Hull MA (HMA), ALMA, KAMA - near-zero lag adaptive filters
- RSI → Ehlers Fisher Transform RSI - normalized, leading
- MACD → Ehlers MESA MACD (Zero-Lag MACD) - adaptive cycle-based
- Stochastic K/D → Ehlers Stochastic - leading stochastic oscillator
- ADX → Ehlers Instantaneous Trendline - trend strength without lag
- Bollinger Bands → Keltner Channels with ATR - volatility bands
- ATR → Ehlers SuperSmoother ATR - reduced noise
- CCI → DEMA-smoothed CCI / Woodies CCI - reduced lag
- ROC → Ehlers Roofing Filter / Acceleration - bandpass filtered momentum
- Williams %R → Keep as-is (already minimal lag)

All indicators are O(1) streaming implementations.
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for streaming indicator suite."""
    # JMA periods for multi-horizon analysis
    jma_periods: tuple = (5, 10, 21, 50, 200)
    jma_phase: float = 0.0  # -100 to 100, 0 = balanced
    jma_power: float = 2.0  # smoothing power

    # Hull Moving Average periods
    hma_periods: tuple = (9, 21, 50)

    # ALMA settings
    alma_period: int = 21
    alma_offset: float = 0.85  # 0-1, typically 0.85
    alma_sigma: float = 6.0  # typically 6

    # KAMA settings
    kama_period: int = 10
    kama_fast_period: int = 2
    kama_slow_period: int = 30

    # Ehlers Fisher RSI
    fisher_rsi_period: int = 14

    # Ehlers MESA MACD
    mesa_fast_limit: float = 0.5
    mesa_slow_limit: float = 0.05

    # Ehlers Stochastic
    ehlers_stoch_period: int = 14

    # Ehlers Instantaneous Trendline
    ehlers_trendline_period: int = 14

    # DEMA-smoothed CCI
    dema_cci_period: int = 20

    # Ehlers Roofing Filter
    roofing_hp_period: int = 48  # High-pass period
    roofing_lp_period: int = 10  # Low-pass period (SuperSmoother)

    # Keltner Channels (ATR-based)
    keltner_period: int = 20
    keltner_multiplier: float = 2.0

    # Ehlers Smoothed True Range
    smooth_atr_period: int = 14

    # Williams %R (kept as-is, minimal lag)
    williams_r_period: int = 14


class JurikMA:
    """
    Jurik Moving Average (JMA) - Near-zero lag adaptive moving average.
    
    JMA uses an adaptive smoothing mechanism that reduces lag while
    maintaining smoothness. It's considered one of the best MA filters.
    
    Parameters:
        period: Lookback period
        phase: Phase shift (-100 to 100), 0 = balanced
        power: Smoothing power (typically 2)
    """
    
    def __init__(self, period: int = 10, phase: float = 0.0, power: float = 2.0):
        self.period = period
        self.phase = phase
        self.power = power
        
        # JMA internal state
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.jma = 0.0
        self.count = 0
        
        # Calculate beta based on period
        self.beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
        
        # Calculate phase ratio
        phase_ratio = phase / 100 + 1.5
        if phase < -100:
            phase_ratio = 0.5
        elif phase > 100:
            phase_ratio = 2.5
        
        self.phase_ratio = phase_ratio
        self.alpha = self.beta ** power
        
    def update(self, price: float) -> Optional[float]:
        """Update JMA with new price, return current value."""
        self.count += 1
        
        if self.count == 1:
            self.e0 = price
            self.e1 = price
            self.e2 = price
            self.jma = price
            return None  # Need warmup
        
        # JMA calculation
        self.e0 = (1 - self.alpha) * price + self.alpha * self.e0
        self.e1 = (price - self.e0) * (1 - self.beta) + self.beta * self.e1
        self.e2 = (self.e0 + self.phase_ratio * self.e1 - self.jma) * \
                  (1 - self.alpha) ** 2 + self.alpha ** 2 * self.e2
        self.jma = self.e2 + self.jma
        
        if self.count >= self.period:
            return self.jma
        return None
    
    def reset(self):
        """Reset state."""
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.jma = 0.0
        self.count = 0


class EhlersFisherRSI:
    """
    Ehlers Fisher Transform RSI - Normalized, leading momentum oscillator.
    
    Applies Fisher Transform to RSI, creating a nearly Gaussian distribution
    with clear turning points. This provides earlier signals than standard RSI.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prices = []
        self.fish = 0.0
        self.prev_fish = 0.0
        
    def update(self, price: float) -> Dict[str, Optional[float]]:
        """Update with new price, return Fisher RSI and signal."""
        self.prices.append(price)
        
        if len(self.prices) < self.period + 1:
            return {'fisher_rsi': None, 'fisher_signal': None}
        
        # Keep only needed prices
        if len(self.prices) > self.period + 1:
            self.prices.pop(0)
        
        # Calculate gains and losses
        gains = 0.0
        losses = 0.0
        
        for i in range(1, len(self.prices)):
            change = self.prices[i] - self.prices[i-1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        if losses == 0:
            rsi = 100.0
        else:
            rs = gains / losses if losses != 0 else 100
            rsi = 100 - (100 / (1 + rs))
        
        # Normalize RSI to -1 to 1 range
        normalized = 0.1 * (rsi - 50) / 50
        
        # Clamp to prevent infinity in Fisher Transform
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher Transform
        self.prev_fish = self.fish
        self.fish = 0.5 * math.log((1 + normalized) / (1 - normalized)) + 0.5 * self.prev_fish
        
        return {
            'fisher_rsi': self.fish,
            'fisher_signal': self.prev_fish
        }
    
    def reset(self):
        """Reset state."""
        self.prices = []
        self.fish = 0.0
        self.prev_fish = 0.0


class EhlersMESAMACD:
    """
    Ehlers MESA Adaptive MACD - Cycle-based adaptive MACD.
    
    Uses Hilbert Transform to measure the dominant cycle and adapts
    the MACD parameters accordingly. Superior to fixed-period MACD.
    """
    
    def __init__(self, fast_limit: float = 0.5, slow_limit: float = 0.05):
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        
        # Internal state
        self.smooth = [0.0] * 7
        self.detrender = [0.0] * 7
        self.q1 = [0.0] * 7
        self.i1 = [0.0] * 7
        self.i2 = 0.0
        self.q2 = 0.0
        self.re = 0.0
        self.im = 0.0
        self.period = 0.0
        self.smooth_period = 0.0
        self.phase = 0.0
        self.mama = 0.0
        self.fama = 0.0
        self.prev_mama = 0.0
        self.prev_fama = 0.0
        self.prices = []
        
    def update(self, price: float) -> Dict[str, Optional[float]]:
        """Update with new price, return MESA MACD values."""
        self.prices.append(price)
        
        if len(self.prices) < 7:
            return {
                'mesa_macd': None,
                'mesa_signal': None,
                'mesa_histogram': None
            }
        
        # Keep limited history
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]
        
        # Weighted moving average (smoothing)
        self.smooth = self.smooth[1:] + [0.0]
        self.smooth[-1] = (4 * price + 3 * self.prices[-2] + 
                          2 * self.prices[-3] + self.prices[-4]) / 10
        
        # Hilbert Transform calculations
        self.detrender = self.detrender[1:] + [0.0]
        self.detrender[-1] = (0.0962 * self.smooth[-1] + 0.5769 * self.smooth[-3] -
                             0.5769 * self.smooth[-5] - 0.0962 * self.smooth[-7]) if len(self.smooth) >= 7 else 0
        
        # Compute InPhase and Quadrature components
        self.q1 = self.q1[1:] + [0.0]
        self.i1 = self.i1[1:] + [0.0]
        
        self.q1[-1] = (0.0962 * self.detrender[-1] + 0.5769 * self.detrender[-3] -
                      0.5769 * self.detrender[-5] - 0.0962 * self.detrender[-7]) if len(self.detrender) >= 7 else 0
        self.i1[-1] = self.detrender[-4] if len(self.detrender) >= 4 else 0
        
        # Advance phase by 90 degrees
        ji = (0.0962 * self.i1[-1] + 0.5769 * self.i1[-3] -
              0.5769 * self.i1[-5] - 0.0962 * self.i1[-7]) if len(self.i1) >= 7 else 0
        jq = (0.0962 * self.q1[-1] + 0.5769 * self.q1[-3] -
              0.5769 * self.q1[-5] - 0.0962 * self.q1[-7]) if len(self.q1) >= 7 else 0
        
        # Phasor addition for 3-bar averaging
        i2_new = self.i1[-1] - jq
        q2_new = self.q1[-1] + ji
        
        # Smooth I and Q components
        self.i2 = 0.2 * i2_new + 0.8 * self.i2
        self.q2 = 0.2 * q2_new + 0.8 * self.q2
        
        # Homodyne Discriminator
        re_new = self.i2 * self.i1[-2] + self.q2 * self.q1[-2] if len(self.i1) >= 2 else 0
        im_new = self.i2 * self.q1[-2] - self.q2 * self.i1[-2] if len(self.i1) >= 2 else 0
        
        self.re = 0.2 * re_new + 0.8 * self.re
        self.im = 0.2 * im_new + 0.8 * self.im
        
        # Compute period
        if self.im != 0 and self.re != 0:
            self.period = 2 * math.pi / math.atan(self.im / self.re)
        
        if self.period > 1.5 * self.smooth_period:
            self.period = 1.5 * self.smooth_period
        if self.period < 0.67 * self.smooth_period:
            self.period = 0.67 * self.smooth_period
        if self.period < 6:
            self.period = 6
        if self.period > 50:
            self.period = 50
        
        self.smooth_period = 0.33 * self.period + 0.67 * self.smooth_period
        
        # Compute phase
        if self.i1[-1] != 0:
            self.phase = math.degrees(math.atan(self.q1[-1] / self.i1[-1]))
        
        # Compute alpha
        delta_phase = max(self.phase - (self.phase if len(self.prices) < 8 else 0), 1)
        alpha = max(self.slow_limit, self.fast_limit / delta_phase)
        
        # MAMA and FAMA
        self.prev_mama = self.mama
        self.prev_fama = self.fama
        self.mama = alpha * price + (1 - alpha) * self.mama
        self.fama = 0.5 * alpha * self.mama + (1 - 0.5 * alpha) * self.fama
        
        # MACD-style output
        macd = self.mama - self.fama
        signal = 0.5 * macd + 0.5 * (self.prev_mama - self.prev_fama)
        histogram = macd - signal
        
        return {
            'mesa_macd': macd,
            'mesa_signal': signal,
            'mesa_histogram': histogram
        }
    
    def reset(self):
        """Reset state."""
        self.__init__(self.fast_limit, self.slow_limit)


class KeltnerChannels:
    """
    Keltner Channels with ATR - Volatility-based bands.
    
    Uses EMA (replaced with JMA) as the middle band and ATR for
    the channel width. More responsive than Bollinger Bands.
    """
    
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier
        self.jma = JurikMA(period)
        self.atr = EhlersSmoothedTR(period)
        
    def update(self, ohlcv: Dict[str, float]) -> Dict[str, Optional[float]]:
        """Update with OHLCV data, return Keltner values."""
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        
        middle = self.jma.update(typical_price)
        atr = self.atr.update(ohlcv)
        
        if middle is None or atr is None:
            return {
                'keltner_upper': None,
                'keltner_middle': None,
                'keltner_lower': None
            }
        
        return {
            'keltner_upper': middle + self.multiplier * atr,
            'keltner_middle': middle,
            'keltner_lower': middle - self.multiplier * atr
        }
    
    def reset(self):
        """Reset state."""
        self.jma.reset()
        self.atr.reset()


class EhlersSmoothedTR:
    """
    Ehlers Smoothed True Range - Reduced noise ATR.
    
    Applies Ehlers SuperSmoother to True Range for cleaner
    volatility measurement with less lag than standard ATR.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_close = None
        self.tr_history = []
        
        # SuperSmoother coefficients
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
        self.c2 = b1
        self.c3 = -a1 * a1
        self.c1 = 1 - self.c2 - self.c3
        
        # State for SuperSmoother
        self.filt = [0.0, 0.0, 0.0]
        
    def update(self, ohlcv: Dict[str, float]) -> Optional[float]:
        """Update with OHLCV, return smoothed ATR."""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        # Calculate True Range
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        
        self.prev_close = close
        self.tr_history.append(tr)
        
        if len(self.tr_history) < 3:
            return None
        
        if len(self.tr_history) > self.period:
            self.tr_history.pop(0)
        
        # Apply SuperSmoother
        self.filt = [self.filt[1], self.filt[2], 0.0]
        self.filt[2] = self.c1 * (tr + self.tr_history[-2]) / 2 + \
                       self.c2 * self.filt[1] + self.c3 * self.filt[0]
        
        if len(self.tr_history) >= self.period:
            return self.filt[2]
        return None
    
    def reset(self):
        """Reset state."""
        self.prev_close = None
        self.tr_history = []
        self.filt = [0.0, 0.0, 0.0]


class HullMovingAverage:
    """
    Hull Moving Average (HMA) - Fast, smooth moving average with minimal lag.

    HMA = WMA(2 * WMA(price, n/2) - WMA(price, n), sqrt(n))

    Invented by Alan Hull, provides excellent smoothing with significantly
    less lag than traditional moving averages.
    """

    def __init__(self, period: int = 21):
        self.period = period
        self.half_period = max(1, period // 2)
        self.sqrt_period = max(1, int(math.sqrt(period)))

        # WMA buffers
        self.prices = []
        self.wma_half_vals = []
        self.count = 0

    def _wma(self, data: list, period: int) -> Optional[float]:
        """Weighted Moving Average calculation."""
        if len(data) < period:
            return None
        recent = data[-period:]
        weights = list(range(1, period + 1))
        return sum(p * w for p, w in zip(recent, weights)) / sum(weights)

    def update(self, price: float) -> Optional[float]:
        """Update HMA with new price."""
        self.count += 1
        self.prices.append(price)

        if len(self.prices) > self.period * 2:
            self.prices = self.prices[-self.period * 2:]

        # Calculate WMA(n) and WMA(n/2)
        wma_full = self._wma(self.prices, self.period)
        wma_half = self._wma(self.prices, self.half_period)

        if wma_full is None or wma_half is None:
            return None

        # Raw HMA value = 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        self.wma_half_vals.append(raw_hma)

        if len(self.wma_half_vals) > self.sqrt_period * 2:
            self.wma_half_vals = self.wma_half_vals[-self.sqrt_period * 2:]

        # Final HMA = WMA(raw_hma, sqrt(n))
        hma = self._wma(self.wma_half_vals, self.sqrt_period)
        return hma

    def reset(self):
        """Reset state."""
        self.prices = []
        self.wma_half_vals = []
        self.count = 0


class ALMA:
    """
    Arnaud Legoux Moving Average (ALMA) - Gaussian-weighted moving average.

    Uses a Gaussian distribution to weight prices, with adjustable offset
    and sigma parameters. Provides excellent smoothing with configurable lag.

    Parameters:
        period: Lookback period
        offset: 0-1, controls the Gaussian peak position (0.85 typical)
        sigma: Controls the Gaussian width (6 typical)
    """

    def __init__(self, period: int = 21, offset: float = 0.85, sigma: float = 6.0):
        self.period = period
        self.offset = offset
        self.sigma = sigma

        # Pre-compute weights
        self.weights = self._compute_weights()
        self.prices = []

    def _compute_weights(self) -> list:
        """Compute Gaussian weights."""
        m = math.floor(self.offset * (self.period - 1))
        s = self.period / self.sigma
        weights = []
        for i in range(self.period):
            w = math.exp(-((i - m) ** 2) / (2 * s * s))
            weights.append(w)
        return weights

    def update(self, price: float) -> Optional[float]:
        """Update ALMA with new price."""
        self.prices.append(price)

        if len(self.prices) > self.period:
            self.prices.pop(0)

        if len(self.prices) < self.period:
            return None

        # Calculate weighted sum
        weighted_sum = sum(p * w for p, w in zip(self.prices, self.weights))
        weight_sum = sum(self.weights)

        return weighted_sum / weight_sum if weight_sum != 0 else None

    def reset(self):
        """Reset state."""
        self.prices = []


class KAMA:
    """
    Kaufman Adaptive Moving Average (KAMA) - Volatility-adaptive moving average.

    Adapts its smoothing based on market volatility. Fast in trends,
    slow in consolidations. One of the best adaptive MAs.

    Parameters:
        period: Efficiency ratio period
        fast_period: Fast EMA period (typically 2)
        slow_period: Slow EMA period (typically 30)
    """

    def __init__(self, period: int = 10, fast_period: int = 2, slow_period: int = 30):
        self.period = period
        self.fast_sc = 2 / (fast_period + 1)  # Fast smoothing constant
        self.slow_sc = 2 / (slow_period + 1)  # Slow smoothing constant

        self.prices = []
        self.kama = None

    def update(self, price: float) -> Optional[float]:
        """Update KAMA with new price."""
        self.prices.append(price)

        if len(self.prices) > self.period + 1:
            self.prices.pop(0)

        if len(self.prices) < self.period + 1:
            return None

        # Calculate Efficiency Ratio (ER)
        change = abs(self.prices[-1] - self.prices[-self.period - 1])
        volatility = sum(abs(self.prices[i] - self.prices[i - 1])
                         for i in range(1, len(self.prices)))

        er = change / volatility if volatility != 0 else 0

        # Calculate smoothing constant (SC)
        sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

        # Update KAMA
        if self.kama is None:
            self.kama = price
        else:
            self.kama = self.kama + sc * (price - self.kama)

        return self.kama

    def reset(self):
        """Reset state."""
        self.prices = []
        self.kama = None


class EhlersStochastic:
    """
    Ehlers Stochastic - Leading stochastic oscillator using Ehlers' methods.

    Applies SuperSmoother filter and normalization for cleaner, leading signals.
    Provides earlier turning point detection than standard Stochastic.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.highs = []
        self.lows = []
        self.closes = []

        # SuperSmoother coefficients
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
        self.c2 = b1
        self.c3 = -a1 * a1
        self.c1 = 1 - self.c2 - self.c3

        # State for SuperSmoother
        self.filt = [0.0, 0.0, 0.0]
        self.stoch = 0.0
        self.signal = 0.0

    def update(self, high: float, low: float, close: float) -> Dict[str, Optional[float]]:
        """Update with HLC data, return Ehlers Stochastic values."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.highs) > self.period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        if len(self.highs) < self.period:
            return {'ehlers_stoch_k': None, 'ehlers_stoch_d': None}

        # Calculate raw stochastic
        highest = max(self.highs)
        lowest = min(self.lows)
        raw_stoch = (close - lowest) / (highest - lowest) if highest != lowest else 0.5

        # Apply SuperSmoother
        self.filt = [self.filt[1], self.filt[2], 0.0]
        self.filt[2] = self.c1 * raw_stoch + self.c2 * self.filt[1] + self.c3 * self.filt[0]

        # Store previous for signal line
        prev_stoch = self.stoch
        self.stoch = self.filt[2]

        # Signal line (smoothed K)
        self.signal = 0.5 * self.stoch + 0.5 * prev_stoch

        return {
            'ehlers_stoch_k': self.stoch * 100,  # Scale to 0-100
            'ehlers_stoch_d': self.signal * 100
        }

    def reset(self):
        """Reset state."""
        self.highs = []
        self.lows = []
        self.closes = []
        self.filt = [0.0, 0.0, 0.0]
        self.stoch = 0.0
        self.signal = 0.0


class EhlersInstantaneousTrendline:
    """
    Ehlers Instantaneous Trendline - Zero-lag trend indicator.

    Uses Hilbert Transform to extract trend without lag.
    Replaces ADX with a more responsive trend strength measure.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.prices = []

        # Hilbert Transform state
        self.smooth = [0.0] * 7
        self.detrender = [0.0] * 7
        self.i1 = [0.0] * 7
        self.q1 = [0.0] * 7
        self.it = 0.0
        self.trendline = 0.0
        self.prev_trendline = 0.0

    def update(self, price: float) -> Dict[str, Optional[float]]:
        """Update with price, return instantaneous trendline."""
        self.prices.append(price)

        if len(self.prices) < 7:
            return {
                'ehlers_trendline': None,
                'ehlers_trend_strength': None,
                'ehlers_trend_direction': None
            }

        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

        # Smooth price using 4-bar weighted average
        self.smooth = self.smooth[1:] + [0.0]
        self.smooth[-1] = (4 * price + 3 * self.prices[-2] +
                          2 * self.prices[-3] + self.prices[-4]) / 10

        # Hilbert Transform for detrender
        self.detrender = self.detrender[1:] + [0.0]
        if len(self.smooth) >= 7:
            self.detrender[-1] = (0.0962 * self.smooth[-1] + 0.5769 * self.smooth[-3] -
                                 0.5769 * self.smooth[-5] - 0.0962 * self.smooth[-7])

        # Compute InPhase and Quadrature
        self.q1 = self.q1[1:] + [0.0]
        self.i1 = self.i1[1:] + [0.0]

        if len(self.detrender) >= 7:
            self.q1[-1] = (0.0962 * self.detrender[-1] + 0.5769 * self.detrender[-3] -
                         0.5769 * self.detrender[-5] - 0.0962 * self.detrender[-7])
        if len(self.detrender) >= 4:
            self.i1[-1] = self.detrender[-4]

        # Instantaneous Trendline
        self.prev_trendline = self.trendline
        self.it = 0.33 * (self.smooth[-1] + 0.5 * (self.smooth[-1] - self.smooth[-2])) + \
                  0.67 * self.it

        alpha = 0.07  # Smoothing factor
        self.trendline = alpha * (2 * self.it - self.smooth[-2]) + (1 - alpha) * self.trendline

        # Calculate trend strength (similar to ADX concept)
        trend_change = abs(self.trendline - self.prev_trendline)
        trend_strength = min(100, trend_change / (abs(price) * 0.0001 + 1e-10) * 100)

        # Trend direction
        trend_direction = 1 if self.trendline > self.prev_trendline else -1

        return {
            'ehlers_trendline': self.trendline,
            'ehlers_trend_strength': trend_strength,
            'ehlers_trend_direction': trend_direction
        }

    def reset(self):
        """Reset state."""
        self.prices = []
        self.smooth = [0.0] * 7
        self.detrender = [0.0] * 7
        self.i1 = [0.0] * 7
        self.q1 = [0.0] * 7
        self.it = 0.0
        self.trendline = 0.0
        self.prev_trendline = 0.0


class DEMACCI:
    """
    DEMA-smoothed CCI - Commodity Channel Index with reduced lag.

    Uses Double Exponential Moving Average instead of SMA for
    smoother, faster signals than traditional CCI.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.typical_prices = []

        # DEMA state (EMA of EMA)
        self.ema1 = None
        self.ema2 = None
        self.alpha = 2 / (period + 1)

        # Mean deviation tracking
        self.dema_history = []

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update with HLC, return DEMA-smoothed CCI."""
        typical_price = (high + low + close) / 3
        self.typical_prices.append(typical_price)

        if len(self.typical_prices) > self.period * 2:
            self.typical_prices.pop(0)

        # Update DEMA
        if self.ema1 is None:
            self.ema1 = typical_price
            self.ema2 = typical_price
        else:
            self.ema1 = self.alpha * typical_price + (1 - self.alpha) * self.ema1
            self.ema2 = self.alpha * self.ema1 + (1 - self.alpha) * self.ema2

        # DEMA = 2 * EMA1 - EMA2
        dema = 2 * self.ema1 - self.ema2
        self.dema_history.append(dema)

        if len(self.dema_history) > self.period:
            self.dema_history.pop(0)

        if len(self.typical_prices) < self.period:
            return None

        # Calculate mean deviation using DEMA as center
        mean_dev = sum(abs(tp - dema) for tp in self.typical_prices[-self.period:]) / self.period

        # CCI formula with DEMA
        if mean_dev == 0:
            return 0.0

        cci = (typical_price - dema) / (0.015 * mean_dev)
        return cci

    def reset(self):
        """Reset state."""
        self.typical_prices = []
        self.ema1 = None
        self.ema2 = None
        self.dema_history = []


class EhlersRoofingFilter:
    """
    Ehlers Roofing Filter - Bandpass filter for momentum detection.

    Combines high-pass and SuperSmoother (low-pass) filters to isolate
    the dominant cycle component. Excellent replacement for ROC.
    """

    def __init__(self, hp_period: int = 48, lp_period: int = 10):
        self.hp_period = hp_period
        self.lp_period = lp_period

        # High-pass filter coefficients
        alpha1 = (math.cos(2 * math.pi / hp_period) +
                  math.sin(2 * math.pi / hp_period) - 1) / \
                  math.cos(2 * math.pi / hp_period)
        self.hp_alpha = alpha1

        # SuperSmoother coefficients
        a1 = math.exp(-1.414 * math.pi / lp_period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / lp_period)
        self.c2 = b1
        self.c3 = -a1 * a1
        self.c1 = 1 - self.c2 - self.c3

        # State
        self.prices = [0.0, 0.0, 0.0]
        self.hp = [0.0, 0.0, 0.0]
        self.filt = [0.0, 0.0, 0.0]

    def update(self, price: float) -> Dict[str, Optional[float]]:
        """Update with price, return roofing filter values."""
        # Shift history
        self.prices = [self.prices[1], self.prices[2], price]

        # High-pass filter
        self.hp = [self.hp[1], self.hp[2], 0.0]
        self.hp[2] = (1 - self.hp_alpha / 2) ** 2 * (self.prices[2] - 2 * self.prices[1] + self.prices[0]) + \
                     2 * (1 - self.hp_alpha) * self.hp[1] - (1 - self.hp_alpha) ** 2 * self.hp[0]

        # SuperSmoother (low-pass)
        self.filt = [self.filt[1], self.filt[2], 0.0]
        self.filt[2] = self.c1 * (self.hp[2] + self.hp[1]) / 2 + \
                       self.c2 * self.filt[1] + self.c3 * self.filt[0]

        # Calculate acceleration (derivative of roofing)
        acceleration = self.filt[2] - self.filt[1]

        return {
            'ehlers_roofing': self.filt[2],
            'ehlers_acceleration': acceleration,
            'ehlers_momentum': np.sign(self.filt[2]) * abs(self.filt[2]) ** 0.5  # Normalized
        }

    def reset(self):
        """Reset state."""
        self.prices = [0.0, 0.0, 0.0]
        self.hp = [0.0, 0.0, 0.0]
        self.filt = [0.0, 0.0, 0.0]


class WilliamsPercentR:
    """
    Williams %R - Keep as-is (already minimal lag).

    Classic oscillator that shows current close relative to the
    highest high over the lookback period. Very responsive.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.highs = []
        self.lows = []

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update with HLC, return Williams %R."""
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > self.period:
            self.highs.pop(0)
            self.lows.pop(0)

        if len(self.highs) < self.period:
            return None

        highest = max(self.highs)
        lowest = min(self.lows)

        if highest == lowest:
            return -50.0  # Neutral

        williams_r = ((highest - close) / (highest - lowest)) * -100
        return williams_r

    def reset(self):
        """Reset state."""
        self.highs = []
        self.lows = []


class StreamingIndicators:
    """
    Advanced O(1) streaming indicator suite with low-lag DSP filters.

    Complete zero-lag indicator replacements:
    - JMA (Jurik Moving Average) instead of EMA
    - HMA (Hull Moving Average) for fast, smooth trends
    - ALMA (Arnaud Legoux Moving Average) for Gaussian-weighted smoothing
    - KAMA (Kaufman Adaptive Moving Average) for volatility-adaptive smoothing
    - Ehlers Fisher RSI instead of standard RSI
    - Ehlers MESA MACD instead of fixed-period MACD
    - Ehlers Stochastic instead of standard Stochastic K/D
    - Ehlers Instantaneous Trendline instead of ADX
    - DEMA-smoothed CCI instead of standard CCI
    - Ehlers Roofing Filter instead of ROC
    - Keltner Channels (ATR-based) instead of Bollinger Bands
    - Ehlers Smoothed True Range instead of standard ATR
    - Williams %R (kept as-is, already minimal lag)

    Example:
        indicators = StreamingIndicators()
        for candle in ohlcv_stream:
            values = indicators.update(candle)
            print(f"JMA21: {values['jma_21']}, Fisher RSI: {values['fisher_rsi']}")
    """
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()

        # Initialize indicator instances
        self.indicators: Dict[str, Any] = {}

        # === MOVING AVERAGES (Top 3 Priority: JMA, HMA, KAMA/ALMA) ===

        # JMA for each period (replacing EMA) - #1 Priority
        for period in self.config.jma_periods:
            self.indicators[f'jma_{period}'] = JurikMA(
                period=period,
                phase=self.config.jma_phase,
                power=self.config.jma_power
            )

        # Hull Moving Average for each period - #3 Priority
        for period in self.config.hma_periods:
            self.indicators[f'hma_{period}'] = HullMovingAverage(period=period)

        # ALMA (Arnaud Legoux Moving Average)
        self.indicators['alma'] = ALMA(
            period=self.config.alma_period,
            offset=self.config.alma_offset,
            sigma=self.config.alma_sigma
        )

        # KAMA (Kaufman Adaptive Moving Average)
        self.indicators['kama'] = KAMA(
            period=self.config.kama_period,
            fast_period=self.config.kama_fast_period,
            slow_period=self.config.kama_slow_period
        )

        # === OSCILLATORS ===

        # Ehlers Fisher RSI (replacing standard RSI) - #2 Priority
        self.indicators['fisher_rsi'] = EhlersFisherRSI(
            period=self.config.fisher_rsi_period
        )

        # Ehlers MESA MACD (Zero-Lag MACD, replacing standard MACD)
        self.indicators['mesa_macd'] = EhlersMESAMACD(
            fast_limit=self.config.mesa_fast_limit,
            slow_limit=self.config.mesa_slow_limit
        )

        # Ehlers Stochastic (replacing standard Stochastic K/D)
        self.indicators['ehlers_stoch'] = EhlersStochastic(
            period=self.config.ehlers_stoch_period
        )

        # DEMA-smoothed CCI (replacing standard CCI)
        self.indicators['dema_cci'] = DEMACCI(
            period=self.config.dema_cci_period
        )

        # Williams %R (kept as-is, already minimal lag)
        self.indicators['williams_r'] = WilliamsPercentR(
            period=self.config.williams_r_period
        )

        # === TREND ===

        # Ehlers Instantaneous Trendline (replacing ADX)
        self.indicators['ehlers_trendline'] = EhlersInstantaneousTrendline(
            period=self.config.ehlers_trendline_period
        )

        # Ehlers Roofing Filter (replacing ROC)
        self.indicators['ehlers_roofing'] = EhlersRoofingFilter(
            hp_period=self.config.roofing_hp_period,
            lp_period=self.config.roofing_lp_period
        )

        # === VOLATILITY ===

        # Keltner Channels (replacing Bollinger Bands)
        self.indicators['keltner'] = KeltnerChannels(
            period=self.config.keltner_period,
            multiplier=self.config.keltner_multiplier
        )

        # Ehlers Smoothed ATR (Ehlers SuperSmoother ATR, replacing standard ATR)
        self.indicators['smooth_atr'] = EhlersSmoothedTR(
            period=self.config.smooth_atr_period
        )

        self.update_count = 0
        logger.info(f"StreamingIndicators initialized with {len(self.indicators)} zero-lag indicators")
        logger.info("  Moving Averages: JMA, HMA, ALMA, KAMA")
        logger.info("  Oscillators: Fisher RSI, MESA MACD, Ehlers Stoch, DEMA CCI, Williams %R")
        logger.info("  Trend: Ehlers Instantaneous Trendline, Ehlers Roofing Filter")
        logger.info("  Volatility: Keltner Channels, Ehlers Smoothed ATR")
    
    def update(self, ohlcv: Dict[str, float]) -> Dict[str, Optional[float]]:
        """
        Update all indicators with new OHLCV data.

        Args:
            ohlcv: Dict with keys 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Dict of current indicator values (None if insufficient history)

        Complexity: O(1) per indicator, O(k) total where k = number of indicators
        """
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        results = {}

        # === MOVING AVERAGES ===

        # JMAs (use close price) - replacing EMA - #1 Priority
        for period in self.config.jma_periods:
            key = f'jma_{period}'
            jma_val = self.indicators[key].update(close)
            results[key] = jma_val
            # Also provide backward-compatible ema_ keys
            results[f'ema_{period}'] = jma_val

        # Hull Moving Averages - #3 Priority
        for period in self.config.hma_periods:
            key = f'hma_{period}'
            hma_val = self.indicators[key].update(close)
            results[key] = hma_val

        # ALMA
        alma_val = self.indicators['alma'].update(close)
        results['alma'] = alma_val

        # KAMA
        kama_val = self.indicators['kama'].update(close)
        results['kama'] = kama_val

        # === OSCILLATORS ===

        # Ehlers Fisher RSI - replacing standard RSI - #2 Priority
        fisher_vals = self.indicators['fisher_rsi'].update(close)
        results['fisher_rsi'] = fisher_vals['fisher_rsi']
        results['fisher_signal'] = fisher_vals['fisher_signal']
        # Backward compatible rsi key
        results['rsi'] = fisher_vals['fisher_rsi']

        # Ehlers MESA MACD (Zero-Lag MACD) - replacing standard MACD
        mesa_vals = self.indicators['mesa_macd'].update(close)
        results['mesa_macd'] = mesa_vals['mesa_macd']
        results['mesa_signal'] = mesa_vals['mesa_signal']
        results['mesa_histogram'] = mesa_vals['mesa_histogram']
        # Backward compatible macd keys
        results['macd_line'] = mesa_vals['mesa_macd']
        results['macd_signal'] = mesa_vals['mesa_signal']
        results['macd_histogram'] = mesa_vals['mesa_histogram']

        # Ehlers Stochastic - replacing standard Stochastic K/D
        stoch_vals = self.indicators['ehlers_stoch'].update(high, low, close)
        results['ehlers_stoch_k'] = stoch_vals['ehlers_stoch_k']
        results['ehlers_stoch_d'] = stoch_vals['ehlers_stoch_d']
        # Backward compatible stochastic keys
        results['stoch_k'] = stoch_vals['ehlers_stoch_k']
        results['stoch_d'] = stoch_vals['ehlers_stoch_d']

        # DEMA-smoothed CCI - replacing standard CCI
        dema_cci_val = self.indicators['dema_cci'].update(high, low, close)
        results['dema_cci'] = dema_cci_val
        # Backward compatible cci key
        results['cci'] = dema_cci_val

        # Williams %R (kept as-is, already minimal lag)
        williams_r_val = self.indicators['williams_r'].update(high, low, close)
        results['williams_r'] = williams_r_val

        # === TREND ===

        # Ehlers Instantaneous Trendline - replacing ADX
        trendline_vals = self.indicators['ehlers_trendline'].update(close)
        results['ehlers_trendline'] = trendline_vals['ehlers_trendline']
        results['ehlers_trend_strength'] = trendline_vals['ehlers_trend_strength']
        results['ehlers_trend_direction'] = trendline_vals['ehlers_trend_direction']
        # Backward compatible adx key
        results['adx'] = trendline_vals['ehlers_trend_strength']

        # Ehlers Roofing Filter - replacing ROC
        roofing_vals = self.indicators['ehlers_roofing'].update(close)
        results['ehlers_roofing'] = roofing_vals['ehlers_roofing']
        results['ehlers_acceleration'] = roofing_vals['ehlers_acceleration']
        results['ehlers_momentum'] = roofing_vals['ehlers_momentum']
        # Backward compatible roc key
        results['roc'] = roofing_vals['ehlers_momentum']

        # === VOLATILITY ===

        # Keltner Channels - replacing Bollinger Bands
        keltner_vals = self.indicators['keltner'].update(ohlcv)
        results['keltner_upper'] = keltner_vals['keltner_upper']
        results['keltner_middle'] = keltner_vals['keltner_middle']
        results['keltner_lower'] = keltner_vals['keltner_lower']
        # Backward compatible bb_ keys
        results['bb_upper'] = keltner_vals['keltner_upper']
        results['bb_middle'] = keltner_vals['keltner_middle']
        results['bb_lower'] = keltner_vals['keltner_lower']

        # Calculate Keltner %B (position within channel, like Bollinger %B)
        if keltner_vals['keltner_upper'] and keltner_vals['keltner_lower']:
            channel_width = keltner_vals['keltner_upper'] - keltner_vals['keltner_lower']
            if channel_width > 0:
                results['keltner_pct_b'] = (close - keltner_vals['keltner_lower']) / channel_width
            else:
                results['keltner_pct_b'] = 0.5
        else:
            results['keltner_pct_b'] = None
        results['bb_pct_b'] = results['keltner_pct_b']  # Backward compatible

        # Ehlers Smoothed ATR (SuperSmoother ATR) - replacing standard ATR
        smooth_atr = self.indicators['smooth_atr'].update(ohlcv)
        results['smooth_atr'] = smooth_atr
        # Backward compatible atr key
        results['atr'] = smooth_atr

        # ATR as percentage of price
        if smooth_atr is not None and close > 0:
            results['atr_pct'] = (smooth_atr / close) * 100
        else:
            results['atr_pct'] = None

        # Add price for convenience
        results['close'] = close
        results['high'] = high
        results['low'] = low

        self.update_count += 1
        return results
    
    def get_all_values(self) -> Dict[str, Optional[float]]:
        """Get current values without updating."""
        # This would require storing last values - simplified for now
        return {}
    
    def reset(self) -> None:
        """Reset all indicators."""
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        self.update_count = 0
        logger.info("StreamingIndicators reset")
