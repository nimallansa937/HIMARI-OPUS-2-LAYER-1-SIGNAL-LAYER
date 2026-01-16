"""
Shadow Environment (Paper Trading)

Before live deployment, strategies must prove themselves in shadow mode:
real market data, simulated execution. Think of it as a probationary
period where the strategy demonstrates it can handle live market
conditions without risking actual capital.

Key metrics:
- Transfer ratio: shadow_sharpe / backtest_sharpe (target: >0.7)
- Execution residuals: prediction error of fill prices
- Regime behavior: consistency across market conditions

Duration: Minimum 2-3 weeks before live deployment
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import logging

from ..core.genome import StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class ShadowTradeRecord:
    """Record of a trade executed in shadow mode."""
    timestamp: datetime
    symbol: str
    direction: int  # 1=long, -1=short
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    strategy_id: str = ""
    execution_slippage: float = 0.0
    holding_duration_hours: float = 0.0

    def close(self, exit_price: float, exit_time: datetime) -> None:
        """Close the trade and calculate PnL."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_time
        self.pnl = self.direction * self.size * (exit_price - self.entry_price)
        self.holding_duration_hours = (exit_time - self.timestamp).total_seconds() / 3600


@dataclass
class ShadowPerformance:
    """Performance metrics from shadow trading."""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    profit_factor: float
    avg_trade_duration_hours: float
    transfer_ratio: float  # KEY METRIC: shadow_sharpe / backtest_sharpe
    execution_residuals: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_pnl': self.total_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'transfer_ratio': self.transfer_ratio,
            'avg_trade_duration_hours': self.avg_trade_duration_hours
        }


class MockMarketDataFeed:
    """Mock market data feed for testing."""

    def __init__(self, volatility: float = 0.02):
        self.volatility = volatility
        self.current_price = 50000  # BTC price

    async def get_day_data(self, day_offset: int) -> List[Dict]:
        """Generate mock daily candles."""
        np.random.seed(42 + day_offset)
        candles = []
        price = self.current_price

        # 24 hourly candles
        for hour in range(24):
            open_price = price
            returns = np.random.randn() * self.volatility
            close_price = price * (1 + returns)
            high = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.005)
            low = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.005)

            candles.append({
                'timestamp': datetime.now() - timedelta(days=-day_offset, hours=-hour),
                'symbol': 'BTCUSDT',
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.uniform(1000, 5000)
            })
            price = close_price

        self.current_price = price
        return candles


class MockExchangeSimulator:
    """Mock exchange for simulating fills."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        latency_ms: float = 10.0
    ):
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms

    def simulate_fill(
        self,
        symbol: str,
        side: str,
        size: float,
        expected_price: float
    ) -> tuple:
        """
        Simulate order fill with slippage.

        Returns:
            (actual_fill_price, residual)
        """
        # Simulate slippage (adverse for taker)
        slippage_mult = self.slippage_bps / 10000
        if side == 'buy':
            actual_price = expected_price * (1 + slippage_mult * np.random.uniform(0.5, 1.5))
        else:
            actual_price = expected_price * (1 - slippage_mult * np.random.uniform(0.5, 1.5))

        residual = actual_price - expected_price
        return actual_price, residual


class ShadowEnvironment:
    """
    Paper trading environment for strategy validation.

    Simulates real trading conditions with:
    - Live market data
    - Realistic execution (slippage, latency)
    - Full position tracking
    - Performance measurement
    """

    def __init__(
        self,
        market_data_feed=None,
        exchange_simulator=None,
        backtest_sharpe: float = 2.0
    ):
        self.market_feed = market_data_feed or MockMarketDataFeed()
        self.exchange = exchange_simulator or MockExchangeSimulator()
        self.backtest_sharpe = backtest_sharpe

        self.trades: List[ShadowTradeRecord] = []
        self.daily_pnl: List[float] = []
        self.execution_residuals: List[float] = []

    async def run_shadow(
        self,
        strategy: StrategyGenome,
        duration_days: int = 21,
        capital: float = 10000
    ) -> ShadowPerformance:
        """
        Run strategy in shadow mode for specified duration.

        Args:
            strategy: Strategy to test
            duration_days: How many days to run
            capital: Starting capital

        Returns:
            ShadowPerformance with all metrics
        """
        start_date = datetime.now()
        position = 0
        entry_record: Optional[ShadowTradeRecord] = None
        equity = capital
        peak_equity = capital
        max_drawdown = 0
        daily_equity = [capital]

        for day in range(duration_days):
            day_pnl = 0
            market_data = await self.market_feed.get_day_data(day)

            for candle in market_data:
                # Extract features and get signal
                features = self._extract_features(candle)
                signal = strategy.evaluate(features)

                # Handle position changes
                if signal != np.sign(position):
                    # Close existing position
                    if position != 0 and entry_record:
                        exit_price = candle['close']
                        entry_record.close(exit_price, candle['timestamp'])
                        day_pnl += entry_record.pnl or 0
                        equity += entry_record.pnl or 0
                        self.trades.append(entry_record)
                        entry_record = None
                        position = 0

                    # Open new position
                    if signal != 0:
                        size = capital * strategy.base_position_pct / candle['close']
                        actual_fill, residual = self.exchange.simulate_fill(
                            symbol=candle['symbol'],
                            side='buy' if signal > 0 else 'sell',
                            size=size,
                            expected_price=candle['close']
                        )
                        self.execution_residuals.append(residual)

                        entry_record = ShadowTradeRecord(
                            timestamp=candle['timestamp'],
                            symbol=candle['symbol'],
                            direction=signal,
                            size=size,
                            entry_price=actual_fill,
                            strategy_id=strategy.id,
                            execution_slippage=residual
                        )
                        position = signal

                # Update drawdown tracking
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

            # Record daily PnL
            self.daily_pnl.append(day_pnl)
            daily_equity.append(equity)

        # Close any remaining position
        if position != 0 and entry_record and market_data:
            last_candle = market_data[-1]
            entry_record.close(last_candle['close'], last_candle['timestamp'])
            equity += entry_record.pnl or 0
            self.trades.append(entry_record)

        # Calculate metrics
        return self._calculate_performance(
            strategy_id=strategy.id,
            start_date=start_date,
            duration_days=duration_days,
            capital=capital,
            final_equity=equity,
            max_drawdown=max_drawdown,
            daily_equity=daily_equity
        )

    def _extract_features(self, candle: Dict) -> np.ndarray:
        """Extract 60-dim feature vector from candle data."""
        features = np.zeros(60, dtype=np.float32)

        # Price features
        features[0] = candle['close']
        features[1] = candle['open']
        features[2] = candle['high']
        features[3] = candle['low']

        # Mock other features for testing
        features[15] = candle['volume']
        features[25] = 50 + np.random.randn() * 15  # RSI
        features[55] = 1  # Regime

        return features

    def _calculate_performance(
        self,
        strategy_id: str,
        start_date: datetime,
        duration_days: int,
        capital: float,
        final_equity: float,
        max_drawdown: float,
        daily_equity: List[float]
    ) -> ShadowPerformance:
        """Calculate comprehensive performance metrics."""

        # Daily returns
        daily_returns = []
        for i in range(1, len(daily_equity)):
            if daily_equity[i-1] > 0:
                ret = (daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1]
                daily_returns.append(ret)

        # Sharpe ratio
        if daily_returns:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns) + 1e-8
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = 0

        # Trade statistics
        closed_trades = [t for t in self.trades if t.pnl is not None]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_duration = (
            np.mean([t.holding_duration_hours for t in closed_trades])
            if closed_trades else 0
        )

        # Transfer ratio
        transfer_ratio = sharpe / self.backtest_sharpe if self.backtest_sharpe > 0 else 0

        return ShadowPerformance(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=start_date + timedelta(days=duration_days),
            total_pnl=final_equity - capital,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            trade_count=len(closed_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_duration_hours=avg_duration,
            transfer_ratio=transfer_ratio,
            execution_residuals=self.execution_residuals.copy(),
            daily_returns=daily_returns
        )

    def reset(self) -> None:
        """Reset environment for new strategy test."""
        self.trades = []
        self.daily_pnl = []
        self.execution_residuals = []


class ShadowPerformanceAnalyzer:
    """Analyze shadow trading results for deployment readiness."""

    def __init__(
        self,
        min_transfer_ratio: float = 0.7,
        max_drawdown: float = 0.15,
        min_trades: int = 20
    ):
        self.min_transfer_ratio = min_transfer_ratio
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades

    def analyze(self, performance: ShadowPerformance) -> Dict[str, Any]:
        """
        Analyze shadow performance for deployment decision.

        Returns:
            Analysis dict with pass/fail status and details
        """
        checks = {
            'transfer_ratio': {
                'value': performance.transfer_ratio,
                'threshold': self.min_transfer_ratio,
                'passed': performance.transfer_ratio >= self.min_transfer_ratio
            },
            'max_drawdown': {
                'value': performance.max_drawdown,
                'threshold': self.max_drawdown,
                'passed': performance.max_drawdown <= self.max_drawdown
            },
            'trade_count': {
                'value': performance.trade_count,
                'threshold': self.min_trades,
                'passed': performance.trade_count >= self.min_trades
            },
            'sharpe_positive': {
                'value': performance.sharpe_ratio,
                'threshold': 0,
                'passed': performance.sharpe_ratio > 0
            }
        }

        all_passed = all(c['passed'] for c in checks.values())

        # Execution quality analysis
        if performance.execution_residuals:
            exec_residuals = np.array(performance.execution_residuals)
            execution_analysis = {
                'mean_slippage': np.mean(exec_residuals),
                'std_slippage': np.std(exec_residuals),
                'max_slippage': np.max(np.abs(exec_residuals)),
                'slippage_consistent': np.std(exec_residuals) < np.mean(np.abs(exec_residuals)) * 2
            }
        else:
            execution_analysis = {}

        return {
            'deployment_ready': all_passed,
            'checks': checks,
            'execution_analysis': execution_analysis,
            'recommendation': 'Deploy' if all_passed else 'Continue shadow or reject'
        }
