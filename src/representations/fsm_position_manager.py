"""
FSM Position Manager - Enhancement 1

Finite State Machine for stateful position management.
Unlike stateless decision trees, FSM tracks current position state
and enables workflows like "scale in gradually" or "hold through noise".

States:
- FLAT: No position, waiting for entry signal
- LONG: Holding long position
- SHORT: Holding short position
- SCALING_IN: Building position incrementally
- SCALING_OUT: Reducing position incrementally
- STOPPED: Recently stopped out, cooldown period
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Tuple, Callable, Optional, List, Any
from datetime import datetime, timedelta
import random


class FSMState(Enum):
    """Position states for the FSM."""
    FLAT = auto()
    LONG = auto()
    SHORT = auto()
    SCALING_IN = auto()
    SCALING_OUT = auto()
    STOPPED = auto()


class Signal(Enum):
    """Input signals from strategy engines."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    STRONG_BUY = auto()
    STRONG_SELL = auto()
    EXIT = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()


class Action(Enum):
    """Output actions for position management."""
    ENTER_LONG = auto()
    ENTER_SHORT = auto()
    ADD_TO_LONG = auto()
    ADD_TO_SHORT = auto()
    EXIT_PARTIAL = auto()
    EXIT_FULL = auto()
    HOLD = auto()
    NO_ACTION = auto()


@dataclass
class ActionResult:
    """Result of FSM processing a signal."""
    action: Action
    size: float  # Position size as fraction (0.0 to 1.0)
    urgency: float  # 0.0 to 1.0, higher = more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardCondition:
    """A condition that must be met for a transition to occur."""
    name: str
    check_func: Callable[['FSMPositionManager', Dict[str, Any]], bool]
    description: str = ""


def time_in_state_guard(min_seconds: float) -> GuardCondition:
    """Create a guard that requires minimum time in current state."""
    def check(manager: 'FSMPositionManager', market_data: Dict[str, Any]) -> bool:
        if manager.state_entry_time is None:
            return True
        elapsed = (datetime.now() - manager.state_entry_time).total_seconds()
        return elapsed >= min_seconds

    return GuardCondition(
        name=f"time_in_state_{min_seconds}s",
        check_func=check,
        description=f"Must be in state for at least {min_seconds} seconds"
    )


def pnl_guard(min_pnl_percent: float) -> GuardCondition:
    """Create a guard that requires minimum P&L percentage."""
    def check(manager: 'FSMPositionManager', market_data: Dict[str, Any]) -> bool:
        if manager.entry_price == 0 or manager.position_size == 0:
            return True
        current_price = market_data.get('close', manager.entry_price)
        if manager.current_state == FSMState.LONG:
            pnl_pct = (current_price - manager.entry_price) / manager.entry_price * 100
        elif manager.current_state == FSMState.SHORT:
            pnl_pct = (manager.entry_price - current_price) / manager.entry_price * 100
        else:
            pnl_pct = 0
        return pnl_pct >= min_pnl_percent

    return GuardCondition(
        name=f"pnl_{min_pnl_percent}pct",
        check_func=check,
        description=f"Must have at least {min_pnl_percent}% profit"
    )


def volatility_guard(max_atr_multiple: float) -> GuardCondition:
    """Create a guard that blocks transitions when volatility is too high."""
    def check(manager: 'FSMPositionManager', market_data: Dict[str, Any]) -> bool:
        atr = market_data.get('atr', 0)
        avg_atr = market_data.get('avg_atr', atr if atr > 0 else 1)
        if avg_atr == 0:
            return True
        atr_multiple = atr / avg_atr
        return atr_multiple <= max_atr_multiple

    return GuardCondition(
        name=f"volatility_{max_atr_multiple}x",
        check_func=check,
        description=f"ATR must be <= {max_atr_multiple}x average"
    )


def cooldown_guard(cooldown_seconds: float) -> GuardCondition:
    """Create a guard that enforces cooldown after stop-out."""
    def check(manager: 'FSMPositionManager', market_data: Dict[str, Any]) -> bool:
        if manager.last_stop_time is None:
            return True
        elapsed = (datetime.now() - manager.last_stop_time).total_seconds()
        return elapsed >= cooldown_seconds

    return GuardCondition(
        name=f"cooldown_{cooldown_seconds}s",
        check_func=check,
        description=f"Must wait {cooldown_seconds}s after stop-out"
    )


@dataclass
class Transition:
    """A state transition definition."""
    from_state: FSMState
    signal: Signal
    to_state: FSMState
    action: Action
    size: float
    urgency: float
    guards: List[GuardCondition] = field(default_factory=list)


class FSMPositionManager:
    """
    Finite State Machine for position management.

    Tracks current position state and determines actions based on
    incoming signals and guard conditions.
    """

    def __init__(self):
        self.current_state: FSMState = FSMState.FLAT
        self.state_entry_time: Optional[datetime] = datetime.now()
        self.position_size: float = 0.0
        self.entry_price: float = 0.0
        self.last_stop_time: Optional[datetime] = None
        self.position_direction: Optional[str] = None  # 'long' or 'short'

        # Build default transition table
        self.transitions: Dict[Tuple[FSMState, Signal], Transition] = {}
        self._build_default_transitions()

    def _build_default_transitions(self):
        """Build the default transition table."""
        # Default guards
        min_time = time_in_state_guard(5.0)  # 5 second minimum
        min_profit = pnl_guard(0.5)  # 0.5% minimum profit for scaling
        low_vol = volatility_guard(2.0)  # Max 2x normal volatility
        cooldown = cooldown_guard(30.0)  # 30 second cooldown after stop

        default_transitions = [
            # From FLAT state
            Transition(FSMState.FLAT, Signal.BUY, FSMState.LONG, Action.ENTER_LONG, 0.5, 0.7, [low_vol]),
            Transition(FSMState.FLAT, Signal.STRONG_BUY, FSMState.LONG, Action.ENTER_LONG, 1.0, 0.9, []),
            Transition(FSMState.FLAT, Signal.SELL, FSMState.SHORT, Action.ENTER_SHORT, 0.5, 0.7, [low_vol]),
            Transition(FSMState.FLAT, Signal.STRONG_SELL, FSMState.SHORT, Action.ENTER_SHORT, 1.0, 0.9, []),
            Transition(FSMState.FLAT, Signal.HOLD, FSMState.FLAT, Action.NO_ACTION, 0.0, 0.0, []),

            # From LONG state
            Transition(FSMState.LONG, Signal.STRONG_BUY, FSMState.SCALING_IN, Action.ADD_TO_LONG, 0.25, 0.6, [min_time, min_profit]),
            Transition(FSMState.LONG, Signal.BUY, FSMState.LONG, Action.HOLD, 0.0, 0.0, []),
            Transition(FSMState.LONG, Signal.HOLD, FSMState.LONG, Action.HOLD, 0.0, 0.0, []),
            Transition(FSMState.LONG, Signal.SELL, FSMState.SCALING_OUT, Action.EXIT_PARTIAL, 0.5, 0.7, [min_time]),
            Transition(FSMState.LONG, Signal.STRONG_SELL, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.9, []),
            Transition(FSMState.LONG, Signal.EXIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.8, []),
            Transition(FSMState.LONG, Signal.STOP_LOSS, FSMState.STOPPED, Action.EXIT_FULL, 1.0, 1.0, []),
            Transition(FSMState.LONG, Signal.TAKE_PROFIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.85, []),

            # From SHORT state
            Transition(FSMState.SHORT, Signal.STRONG_SELL, FSMState.SCALING_IN, Action.ADD_TO_SHORT, 0.25, 0.6, [min_time, min_profit]),
            Transition(FSMState.SHORT, Signal.SELL, FSMState.SHORT, Action.HOLD, 0.0, 0.0, []),
            Transition(FSMState.SHORT, Signal.HOLD, FSMState.SHORT, Action.HOLD, 0.0, 0.0, []),
            Transition(FSMState.SHORT, Signal.BUY, FSMState.SCALING_OUT, Action.EXIT_PARTIAL, 0.5, 0.7, [min_time]),
            Transition(FSMState.SHORT, Signal.STRONG_BUY, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.9, []),
            Transition(FSMState.SHORT, Signal.EXIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.8, []),
            Transition(FSMState.SHORT, Signal.STOP_LOSS, FSMState.STOPPED, Action.EXIT_FULL, 1.0, 1.0, []),
            Transition(FSMState.SHORT, Signal.TAKE_PROFIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.85, []),

            # From SCALING_IN state
            Transition(FSMState.SCALING_IN, Signal.STRONG_BUY, FSMState.SCALING_IN, Action.ADD_TO_LONG, 0.25, 0.5, [min_time, min_profit]),
            Transition(FSMState.SCALING_IN, Signal.STRONG_SELL, FSMState.SCALING_IN, Action.ADD_TO_SHORT, 0.25, 0.5, [min_time, min_profit]),
            Transition(FSMState.SCALING_IN, Signal.HOLD, FSMState.LONG, Action.HOLD, 0.0, 0.0, []),  # Return to base state
            Transition(FSMState.SCALING_IN, Signal.EXIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.8, []),
            Transition(FSMState.SCALING_IN, Signal.STOP_LOSS, FSMState.STOPPED, Action.EXIT_FULL, 1.0, 1.0, []),

            # From SCALING_OUT state
            Transition(FSMState.SCALING_OUT, Signal.SELL, FSMState.SCALING_OUT, Action.EXIT_PARTIAL, 0.25, 0.6, [min_time]),
            Transition(FSMState.SCALING_OUT, Signal.BUY, FSMState.SCALING_OUT, Action.EXIT_PARTIAL, 0.25, 0.6, [min_time]),
            Transition(FSMState.SCALING_OUT, Signal.STRONG_SELL, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.9, []),
            Transition(FSMState.SCALING_OUT, Signal.STRONG_BUY, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.9, []),
            Transition(FSMState.SCALING_OUT, Signal.HOLD, FSMState.LONG, Action.HOLD, 0.0, 0.0, []),  # Return to holding
            Transition(FSMState.SCALING_OUT, Signal.EXIT, FSMState.FLAT, Action.EXIT_FULL, 1.0, 0.8, []),
            Transition(FSMState.SCALING_OUT, Signal.STOP_LOSS, FSMState.STOPPED, Action.EXIT_FULL, 1.0, 1.0, []),

            # From STOPPED state (cooldown)
            Transition(FSMState.STOPPED, Signal.BUY, FSMState.LONG, Action.ENTER_LONG, 0.25, 0.5, [cooldown, low_vol]),
            Transition(FSMState.STOPPED, Signal.STRONG_BUY, FSMState.LONG, Action.ENTER_LONG, 0.5, 0.6, [cooldown]),
            Transition(FSMState.STOPPED, Signal.SELL, FSMState.SHORT, Action.ENTER_SHORT, 0.25, 0.5, [cooldown, low_vol]),
            Transition(FSMState.STOPPED, Signal.STRONG_SELL, FSMState.SHORT, Action.ENTER_SHORT, 0.5, 0.6, [cooldown]),
            Transition(FSMState.STOPPED, Signal.HOLD, FSMState.FLAT, Action.NO_ACTION, 0.0, 0.0, [cooldown]),
        ]

        for t in default_transitions:
            self.transitions[(t.from_state, t.signal)] = t

    def add_transition(self, transition: Transition):
        """Add or update a transition in the table."""
        self.transitions[(transition.from_state, transition.signal)] = transition

    def process_signal(
        self,
        signal: Signal,
        market_data: Dict[str, Any],
        confidence: float = 1.0
    ) -> ActionResult:
        """
        Process an incoming signal and determine the action.

        Args:
            signal: The signal from the strategy engine
            market_data: Current market data for guard evaluation
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            ActionResult with action, size, and urgency
        """
        key = (self.current_state, signal)

        # Check if transition exists
        if key not in self.transitions:
            return ActionResult(
                action=Action.NO_ACTION,
                size=0.0,
                urgency=0.0,
                metadata={"reason": "no_transition_defined"}
            )

        transition = self.transitions[key]

        # Check all guard conditions
        failed_guards = []
        for guard in transition.guards:
            if not guard.check_func(self, market_data):
                failed_guards.append(guard.name)

        if failed_guards:
            return ActionResult(
                action=Action.NO_ACTION,
                size=0.0,
                urgency=0.0,
                metadata={"reason": "guards_failed", "failed_guards": failed_guards}
            )

        # Execute transition
        old_state = self.current_state
        self.current_state = transition.to_state
        self.state_entry_time = datetime.now()

        # Update position tracking
        if transition.action == Action.ENTER_LONG:
            self.position_size = transition.size * confidence
            self.entry_price = market_data.get('close', 0)
            self.position_direction = 'long'
        elif transition.action == Action.ENTER_SHORT:
            self.position_size = transition.size * confidence
            self.entry_price = market_data.get('close', 0)
            self.position_direction = 'short'
        elif transition.action in (Action.ADD_TO_LONG, Action.ADD_TO_SHORT):
            self.position_size = min(1.0, self.position_size + transition.size * confidence)
        elif transition.action == Action.EXIT_PARTIAL:
            self.position_size = max(0.0, self.position_size - transition.size)
            if self.position_size == 0:
                self.current_state = FSMState.FLAT
        elif transition.action == Action.EXIT_FULL:
            self.position_size = 0.0
            self.entry_price = 0.0
            self.position_direction = None
            if transition.to_state == FSMState.STOPPED:
                self.last_stop_time = datetime.now()

        return ActionResult(
            action=transition.action,
            size=transition.size * confidence,
            urgency=transition.urgency * confidence,
            metadata={
                "from_state": old_state.name,
                "to_state": self.current_state.name,
                "position_size": self.position_size,
                "entry_price": self.entry_price
            }
        )

    def reset(self):
        """Reset FSM to initial state."""
        self.current_state = FSMState.FLAT
        self.state_entry_time = datetime.now()
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_stop_time = None
        self.position_direction = None

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "state": self.current_state.name,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_direction": self.position_direction,
            "time_in_state": (datetime.now() - self.state_entry_time).total_seconds() if self.state_entry_time else 0,
            "last_stop_time": self.last_stop_time.isoformat() if self.last_stop_time else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FSM configuration to dictionary."""
        return {
            "current_state": self.current_state.name,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "position_direction": self.position_direction,
            "transitions": {
                f"{k[0].name}_{k[1].name}": {
                    "to_state": v.to_state.name,
                    "action": v.action.name,
                    "size": v.size,
                    "urgency": v.urgency,
                    "guards": [g.name for g in v.guards]
                }
                for k, v in self.transitions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FSMPositionManager':
        """Deserialize FSM from dictionary."""
        manager = cls()
        manager.current_state = FSMState[data.get("current_state", "FLAT")]
        manager.position_size = data.get("position_size", 0.0)
        manager.entry_price = data.get("entry_price", 0.0)
        manager.position_direction = data.get("position_direction")
        return manager


# Genetic operators for FSM evolution
class FSMGeneticOperators:
    """Genetic operators for evolving FSM configurations."""

    @staticmethod
    def mutate_transition(manager: FSMPositionManager) -> FSMPositionManager:
        """Randomly change one transition's target state or action."""
        if not manager.transitions:
            return manager

        key = random.choice(list(manager.transitions.keys()))
        transition = manager.transitions[key]

        mutation_type = random.choice(['state', 'action', 'size', 'urgency'])

        if mutation_type == 'state':
            new_state = random.choice(list(FSMState))
            manager.transitions[key] = Transition(
                transition.from_state,
                transition.signal,
                new_state,
                transition.action,
                transition.size,
                transition.urgency,
                transition.guards
            )
        elif mutation_type == 'action':
            new_action = random.choice(list(Action))
            manager.transitions[key] = Transition(
                transition.from_state,
                transition.signal,
                transition.to_state,
                new_action,
                transition.size,
                transition.urgency,
                transition.guards
            )
        elif mutation_type == 'size':
            new_size = max(0.0, min(1.0, transition.size + random.uniform(-0.2, 0.2)))
            manager.transitions[key] = Transition(
                transition.from_state,
                transition.signal,
                transition.to_state,
                transition.action,
                new_size,
                transition.urgency,
                transition.guards
            )
        else:  # urgency
            new_urgency = max(0.0, min(1.0, transition.urgency + random.uniform(-0.2, 0.2)))
            manager.transitions[key] = Transition(
                transition.from_state,
                transition.signal,
                transition.to_state,
                transition.action,
                transition.size,
                new_urgency,
                transition.guards
            )

        return manager

    @staticmethod
    def mutate_guard(manager: FSMPositionManager, threshold_change: float = 0.1) -> FSMPositionManager:
        """Adjust guard threshold by ±10%."""
        # Note: This would need to recreate guards with adjusted thresholds
        # For now, we modify the transition's size/urgency as a proxy
        if not manager.transitions:
            return manager

        key = random.choice(list(manager.transitions.keys()))
        transition = manager.transitions[key]

        factor = 1.0 + random.uniform(-threshold_change, threshold_change)
        new_size = max(0.0, min(1.0, transition.size * factor))
        new_urgency = max(0.0, min(1.0, transition.urgency * factor))

        manager.transitions[key] = Transition(
            transition.from_state,
            transition.signal,
            transition.to_state,
            transition.action,
            new_size,
            new_urgency,
            transition.guards
        )

        return manager

    @staticmethod
    def crossover_fsm(parent1: FSMPositionManager, parent2: FSMPositionManager) -> Tuple[FSMPositionManager, FSMPositionManager]:
        """Swap transition subsets between two FSMs."""
        child1 = FSMPositionManager()
        child2 = FSMPositionManager()

        all_keys = set(parent1.transitions.keys()) | set(parent2.transitions.keys())

        for key in all_keys:
            if random.random() < 0.5:
                if key in parent1.transitions:
                    child1.transitions[key] = parent1.transitions[key]
                if key in parent2.transitions:
                    child2.transitions[key] = parent2.transitions[key]
            else:
                if key in parent2.transitions:
                    child1.transitions[key] = parent2.transitions[key]
                if key in parent1.transitions:
                    child2.transitions[key] = parent1.transitions[key]

        return child1, child2
