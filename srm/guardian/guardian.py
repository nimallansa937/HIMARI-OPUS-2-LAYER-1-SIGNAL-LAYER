"""
Systemic Risk Guardian

Translates SRM risk scores into trading constraints.
Implements tiered response system with emergency exit functionality.

Response Tiers:
- Score < 0.5: Normal operation, full position sizing
- Score 0.5-0.7: Reduced exposure, half position sizes
- Score 0.7-0.9: Close-only mode, no new positions
- Score > 0.9: Emergency halt, liquidate all positions
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import logging
import asyncio

from ..services import SRMRedisClient

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Trading actions based on risk level."""
    NORMAL = "normal"          # Full position sizing allowed
    REDUCE = "reduce"          # Half position sizes
    CLOSE_ONLY = "close_only"  # Only closing positions allowed
    HALT = "halt"              # No trading, liquidate existing


@dataclass
class RiskDecision:
    """Decision output from SystemicRiskGuardian."""
    action: RiskAction
    position_multiplier: float  # 0.0-1.0 scale factor for position sizes
    reason: str
    requires_emergency_exit: bool
    metadata: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'position_multiplier': self.position_multiplier,
            'reason': self.reason,
            'requires_emergency_exit': self.requires_emergency_exit,
            'metadata': self.metadata
        }


class EmergencyExitExecutor:
    """
    Emergency position liquidation via Binance Futures.
    
    Uses market orders for immediate execution with reduceOnly flag.
    Speed is critical—October 10, 2025 cascade accelerated from
    $0 to $19B liquidated in under 1 hour.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        dry_run: bool = True
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self._client = None
        
        if not dry_run and api_key and api_secret:
            try:
                from binance.um_futures import UMFutures
                self._client = UMFutures(key=api_key, secret=api_secret)
                logger.info("Binance Futures client initialized for emergency exits")
            except ImportError:
                logger.warning(
                    "binance-futures-connector not installed. "
                    "Run: pip install binance-futures-connector"
                )
    
    async def close_position(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> dict:
        """
        Emergency market order with reduce_only flag.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: 'BUY' to close short, 'SELL' to close long
            quantity: Position size to close
        
        Returns:
            Order result dict
        """
        if self.dry_run or self._client is None:
            logger.critical(
                f"[DRY RUN] Would close {quantity} {symbol} via {side}"
            )
            return {
                'status': 'dry_run',
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }
        
        try:
            # Execute market order with reduceOnly flag
            # reduceOnly=True is CRITICAL: ensures it closes, not opens
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.new_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity,
                    reduceOnly=True
                )
            )
            
            logger.critical(
                f"EMERGENCY EXIT EXECUTED: {side} {quantity} {symbol} "
                f"orderId={result.get('orderId')}"
            )
            
            return {
                'status': 'executed',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_id': result.get('orderId'),
                'fill_price': result.get('avgPrice')
            }
            
        except Exception as e:
            logger.error(f"Emergency exit failed for {symbol}: {e}")
            return {
                'status': 'failed',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'error': str(e)
            }
    
    async def get_positions(self) -> List[dict]:
        """
        Get current open positions.
        
        Returns:
            List of position dicts with symbol, side, quantity
        
        Raises:
            No exceptions - returns empty list on failure with error logged
        """
        if self._client is None:
            logger.debug("No Binance client configured - returning empty positions (dry-run mode)")
            return []
        
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.get_position_risk()
            )
            
            if positions is None:
                logger.warning("get_position_risk returned None")
                return []
            
            # Filter to open positions only
            open_positions = []
            for pos in positions:
                try:
                    amt = float(pos.get('positionAmt', 0))
                    if amt != 0:
                        open_positions.append({
                            'symbol': pos['symbol'],
                            'side': 'LONG' if amt > 0 else 'SHORT',
                            'quantity': abs(amt),
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'unrealized_pnl': float(pos.get('unRealizedProfit', 0))
                        })
                except (KeyError, ValueError, TypeError) as parse_err:
                    logger.warning(f"Failed to parse position data: {parse_err}")
                    continue
            
            return open_positions
            
        except ConnectionError as e:
            logger.error(f"Network error getting positions (check connectivity): {e}")
            return []
        except PermissionError as e:
            logger.error(f"Authentication error getting positions (check API keys): {e}")
            return []
        except Exception as e:
            # Handle Binance-specific error codes
            error_str = str(e)
            if 'APIError' in error_str or '-' in error_str[:10]:
                logger.error(f"Binance API error getting positions: {e}")
            else:
                logger.error(f"Unexpected error getting positions: {type(e).__name__}: {e}")
            return []


class SystemicRiskGuardian:
    """
    Translates SRM risk scores into trading constraints.
    
    The Guardian implements a tiered response system:
    - Score < 0.5: Normal operation, full position sizing
    - Score 0.5-0.7: Reduced exposure, half position sizes
    - Score 0.7-0.9: Close-only mode, no new positions
    - Score > 0.9: Emergency halt, liquidate all positions
    
    Additionally monitors risk velocity—a sudden spike (>0.3 in 5 minutes)
    triggers emergency response regardless of absolute level.
    """
    
    def __init__(
        self,
        redis_client: Optional[SRMRedisClient] = None,
        emergency_executor: Optional[EmergencyExitExecutor] = None,
        reduce_threshold: float = 0.5,
        close_only_threshold: float = 0.7,
        halt_threshold: float = 0.9,
        confirmation_count: int = 3,
        velocity_threshold: float = 0.001  # Score/second
    ):
        self.redis = redis_client
        self.executor = emergency_executor
        
        # Thresholds
        self.reduce_threshold = reduce_threshold
        self.close_only_threshold = close_only_threshold
        self.halt_threshold = halt_threshold
        self.confirmation_count = confirmation_count
        self.velocity_threshold = velocity_threshold
        
        # State
        self.consecutive_critical_count = 0
        self._last_decision: Optional[RiskDecision] = None
    
    def evaluate(self, symbol: str) -> RiskDecision:
        """
        Evaluate current risk and return trading decision.
        
        Args:
            symbol: Trading symbol, e.g., 'BTCUSDT'
        
        Returns:
            RiskDecision with action and position multiplier
        """
        risk_data = None
        if self.redis:
            risk_data = self.redis.get_current_risk(symbol)
        
        if risk_data is None:
            # Stale or missing data - conservative response
            logger.warning(f"No current risk data for {symbol}")
            decision = RiskDecision(
                action=RiskAction.REDUCE,
                position_multiplier=0.5,
                reason="Risk data unavailable - operating conservatively",
                requires_emergency_exit=False,
                metadata={'status': 'no_data'}
            )
            self._last_decision = decision
            return decision
        
        score = risk_data['score']
        regime = risk_data['regime']
        
        # Check velocity (rapid escalation detection)
        velocity = None
        if self.redis:
            velocity = self.redis.get_risk_velocity(symbol, window_seconds=300)
        velocity_critical = velocity is not None and velocity > self.velocity_threshold
        
        # Tier 1: Emergency halt (score > 0.9 or velocity spike)
        if score > self.halt_threshold or velocity_critical:
            self.consecutive_critical_count += 1
            
            if self.consecutive_critical_count >= self.confirmation_count:
                logger.critical(
                    f"EMERGENCY HALT: {symbol} score={score:.3f} "
                    f"velocity={velocity:.6f if velocity else 0}/s regime={regime}"
                )
                decision = RiskDecision(
                    action=RiskAction.HALT,
                    position_multiplier=0.0,
                    reason=f"Critical risk: score={score:.3f}, velocity={velocity}",
                    requires_emergency_exit=True,
                    metadata={
                        'score': score,
                        'velocity': velocity,
                        'regime': regime,
                        'consecutive_critical': self.consecutive_critical_count
                    }
                )
                self._last_decision = decision
                return decision
            else:
                # Awaiting confirmation
                decision = RiskDecision(
                    action=RiskAction.CLOSE_ONLY,
                    position_multiplier=0.0,
                    reason=f"Critical risk pending confirmation ({self.consecutive_critical_count}/{self.confirmation_count})",
                    requires_emergency_exit=False,
                    metadata={'score': score, 'awaiting_confirmation': True}
                )
                self._last_decision = decision
                return decision
        
        # Reset confirmation counter if not critical
        self.consecutive_critical_count = max(0, self.consecutive_critical_count - 1)
        
        # Tier 2: Close-only (score 0.7-0.9)
        if score > self.close_only_threshold:
            logger.warning(f"HIGH RISK: {symbol} score={score:.3f} - close-only mode")
            decision = RiskDecision(
                action=RiskAction.CLOSE_ONLY,
                position_multiplier=0.0,
                reason=f"High risk: score={score:.3f}",
                requires_emergency_exit=False,
                metadata={'score': score, 'regime': regime}
            )
            self._last_decision = decision
            return decision
        
        # Tier 3: Reduced exposure (score 0.5-0.7)
        if score > self.reduce_threshold:
            logger.info(f"ELEVATED RISK: {symbol} score={score:.3f} - reducing exposure")
            decision = RiskDecision(
                action=RiskAction.REDUCE,
                position_multiplier=0.5,
                reason=f"Elevated risk: score={score:.3f}",
                requires_emergency_exit=False,
                metadata={'score': score, 'regime': regime}
            )
            self._last_decision = decision
            return decision
        
        # Tier 4: Normal operation
        decision = RiskDecision(
            action=RiskAction.NORMAL,
            position_multiplier=1.0,
            reason="Normal risk conditions",
            requires_emergency_exit=False,
            metadata={'score': score, 'regime': regime}
        )
        self._last_decision = decision
        return decision
    
    async def execute_emergency_exit(self, symbol: str) -> dict:
        """
        Execute emergency position liquidation.
        
        Uses market orders for immediate execution.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict with execution results
        """
        if not self.executor:
            logger.error("No emergency executor configured")
            return {'status': 'no_executor', 'positions_closed': 0}
        
        # Get current positions
        positions = await self.executor.get_positions()
        
        if not positions:
            logger.info("No open positions to close")
            return {'status': 'no_positions', 'positions_closed': 0}
        
        results = []
        for position in positions:
            if position['symbol'] == symbol or symbol == 'ALL':
                # Determine closing side
                close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
                
                result = await self.executor.close_position(
                    symbol=position['symbol'],
                    side=close_side,
                    quantity=position['quantity']
                )
                
                results.append({
                    'position': position['symbol'],
                    'side': close_side,
                    'quantity': position['quantity'],
                    **result
                })
        
        # Publish alert
        if self.redis:
            self.redis.publish_alert(symbol, {
                'type': 'EMERGENCY_EXIT',
                'positions_closed': len(results),
                'results': results
            })
        
        logger.critical(
            f"🚨 EMERGENCY EXIT COMPLETE: {symbol} - {len(results)} positions closed"
        )
        
        return {
            'status': 'executed',
            'positions_closed': len(results),
            'results': results
        }
    
    @property
    def last_decision(self) -> Optional[RiskDecision]:
        """Get last decision made."""
        return self._last_decision
