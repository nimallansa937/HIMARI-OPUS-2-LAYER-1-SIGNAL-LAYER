"""
Transaction Cost Model for HIMARI L1

Models realistic trading costs to filter out strategies that
cannot survive friction:
- Commission: 0.1-0.25% per side
- Bid-ask spread: 2-8 bp (crypto) vs 0.5-2 bp (equities)
- Slippage: 10-100 bp during volatility
- Market impact: Size-dependent

CRITICAL: Transaction costs consume 30%+ of gross returns.
Model 20-25 bp total per trade for crypto.

Usage:
    cost_model = TransactionCostModel(asset_class='crypto')
    
    # Evaluate strategy profitability after costs
    net_return = cost_model.apply_costs(
        gross_return=0.02,
        n_trades=10,
        avg_trade_size=1000
    )
"""

import math
from typing import Dict, Any, Optional


class TransactionCostModel:
    """
    Comprehensive transaction cost model.
    
    Includes all cost components: commission, spread, slippage, impact.
    """
    
    def __init__(
        self,
        asset_class: str = 'crypto',
        commission_rate: Optional[float] = None,
        spread_bps: Optional[float] = None,
        base_slippage_bps: Optional[float] = None
    ):
        """
        Initialize transaction cost model.
        
        Args:
            asset_class: 'crypto', 'equity', or 'futures'
            commission_rate: Override default commission
            spread_bps: Override default spread in basis points
            base_slippage_bps: Override default slippage
        """
        self.asset_class = asset_class
        
        # Set defaults based on asset class
        defaults = {
            'crypto': {
                'commission': 0.001,  # 0.1% (e.g., Binance maker)
                'spread_bps': 5,      # 5 basis points
                'slippage_bps': 10,   # 10 basis points base
            },
            'equity': {
                'commission': 0.0001,  # 0.01% (near zero for retail)
                'spread_bps': 1,       # 1 basis point
                'slippage_bps': 5,     # 5 basis points base
            },
            'futures': {
                'commission': 0.0002,  # 0.02%
                'spread_bps': 0.5,     # 0.5 basis points
                'slippage_bps': 3,     # 3 basis points base
            }
        }
        
        config = defaults.get(asset_class, defaults['crypto'])
        
        self.commission_rate = commission_rate or config['commission']
        self.spread_bps = spread_bps or config['spread_bps']
        self.base_slippage_bps = base_slippage_bps or config['slippage_bps']
    
    def cost_per_trade(
        self,
        trade_size_usd: float = 1000,
        volatility: float = 0.02,
        is_taker: bool = True
    ) -> Dict[str, float]:
        """
        Compute all-in cost for a single trade.
        
        Args:
            trade_size_usd: Trade size in USD
            volatility: Current daily volatility
            is_taker: True if market order (pays spread), False if limit
            
        Returns:
            Dict with cost breakdown in basis points and dollars
        """
        # Commission (both sides)
        commission_bps = self.commission_rate * 10000 * 2
        
        # Spread (only for taker orders)
        spread_cost_bps = self.spread_bps if is_taker else 0
        
        # Slippage (increases with volatility)
        # Volatility multiplier: 1x at 2% vol, 2x at 4% vol, etc.
        vol_multiplier = max(1, volatility / 0.02)
        slippage_bps = self.base_slippage_bps * vol_multiplier
        
        # Market impact (increases with square root of size)
        # Impact = base * sqrt(size / reference_size)
        reference_size = 10000  # $10k reference
        impact_bps = 1 * math.sqrt(trade_size_usd / reference_size)
        
        # Total
        total_bps = commission_bps + spread_cost_bps + slippage_bps + impact_bps
        total_dollars = (total_bps / 10000) * trade_size_usd
        
        return {
            'commission_bps': commission_bps,
            'spread_bps': spread_cost_bps,
            'slippage_bps': slippage_bps,
            'impact_bps': impact_bps,
            'total_bps': total_bps,
            'total_dollars': total_dollars,
            'total_percent': total_bps / 100,
        }
    
    def apply_costs(
        self,
        gross_return: float,
        n_trades: int,
        avg_trade_size: float = 1000,
        avg_volatility: float = 0.02
    ) -> Dict[str, float]:
        """
        Apply transaction costs to gross return.
        
        Args:
            gross_return: Gross return (e.g., 0.10 = 10%)
            n_trades: Number of round-trip trades
            avg_trade_size: Average trade size in USD
            avg_volatility: Average volatility during period
            
        Returns:
            Dict with net return and cost breakdown
        """
        # Cost per trade
        cost_per = self.cost_per_trade(
            avg_trade_size, avg_volatility, is_taker=True
        )
        
        # Total costs
        total_cost_pct = (cost_per['total_bps'] / 10000) * n_trades
        
        # Net return
        net_return = gross_return - total_cost_pct
        
        return {
            'gross_return': gross_return,
            'total_cost_pct': total_cost_pct,
            'net_return': net_return,
            'cost_per_trade_bps': cost_per['total_bps'],
            'n_trades': n_trades,
            'cost_drag_pct': (total_cost_pct / gross_return * 100) 
                            if gross_return > 0 else 0,
        }
    
    def minimum_gross_sharpe(
        self,
        annual_turnover: float = 12,  # 12x = monthly rebalance
        avg_trade_size: float = 1000,
        target_net_sharpe: float = 1.5
    ) -> float:
        """
        Compute minimum gross Sharpe required for target net Sharpe.
        
        Args:
            annual_turnover: Number of round-trips per year
            avg_trade_size: Average trade size
            target_net_sharpe: Desired net Sharpe after costs
            
        Returns:
            Minimum required gross Sharpe ratio
        """
        # Approximate annual cost drag
        cost_per = self.cost_per_trade(avg_trade_size, 0.02)
        annual_cost_pct = (cost_per['total_bps'] / 10000) * annual_turnover
        
        # Assuming 15% annual volatility baseline
        vol = 0.15
        
        # Cost Sharpe impact = cost / vol
        cost_sharpe_drag = annual_cost_pct / vol
        
        return target_net_sharpe + cost_sharpe_drag
    
    def validate_strategy(
        self,
        gross_sharpe: float,
        annual_turnover: float,
        avg_trade_size: float = 1000,
        min_net_sharpe: float = 1.5
    ) -> Dict[str, Any]:
        """
        Validate whether strategy survives transaction costs.
        
        Args:
            gross_sharpe: Backtest Sharpe ratio
            annual_turnover: Number of round-trips per year
            avg_trade_size: Average trade size
            min_net_sharpe: Minimum acceptable net Sharpe
            
        Returns:
            Validation result with breakdown
        """
        # Compute cost drag
        cost_per = self.cost_per_trade(avg_trade_size, 0.02)
        annual_cost_pct = (cost_per['total_bps'] / 10000) * annual_turnover
        
        # Estimate net Sharpe (simplified)
        vol = 0.15  # Assumed 15% annual vol
        cost_sharpe_drag = annual_cost_pct / vol
        net_sharpe = gross_sharpe - cost_sharpe_drag
        
        passed = net_sharpe >= min_net_sharpe
        
        return {
            'passed': passed,
            'gross_sharpe': gross_sharpe,
            'net_sharpe': net_sharpe,
            'cost_sharpe_drag': cost_sharpe_drag,
            'annual_cost_pct': annual_cost_pct * 100,
            'cost_per_trade_bps': cost_per['total_bps'],
            'annual_turnover': annual_turnover,
            'reason': 'passed' if passed else 'costs_too_high',
        }
    
    def __repr__(self) -> str:
        cost = self.cost_per_trade(1000, 0.02)
        return (
            f"TransactionCostModel({self.asset_class}, "
            f"~{cost['total_bps']:.1f}bp/trade)"
        )


class TransactionCostOptimizer:
    """
    Utility for optimizing trade execution to minimize costs.
    """
    
    @staticmethod
    def optimal_order_size(
        target_size: float,
        avg_daily_volume: float,
        max_participation: float = 0.1
    ) -> int:
        """
        Compute optimal number of child orders to minimize impact.
        
        Args:
            target_size: Total size to execute
            avg_daily_volume: Average daily volume
            max_participation: Max participation rate
            
        Returns:
            Number of child orders recommended
        """
        max_per_order = avg_daily_volume * max_participation
        
        if target_size <= max_per_order:
            return 1
        
        return max(1, int(math.ceil(target_size / max_per_order)))
    
    @staticmethod
    def maker_vs_taker_breakeven(
        model: TransactionCostModel,
        price_improvement_prob: float = 0.5
    ) -> float:
        """
        Compute how long to wait for limit order fill to beat market order.
        
        Returns time in seconds that limit order has to fill to break even
        with immediate market order.
        """
        # Cost savings from maker vs taker
        savings_bps = model.spread_bps
        
        # If limit order doesn't fill, opportunity cost depends on
        # expected price movement during wait time
        # Simplified: assume ~5 bps per minute of adverse movement
        
        breakeven_minutes = savings_bps / 5 * (1 / (1 - price_improvement_prob))
        
        return breakeven_minutes * 60  # Return seconds
