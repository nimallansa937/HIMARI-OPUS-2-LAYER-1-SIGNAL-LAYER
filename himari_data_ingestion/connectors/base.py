"""
Base Connector Abstract Class

All exchange connectors inherit from this class to ensure consistent:
- Message callback handling
- Reconnection logic with exponential backoff
- Health monitoring
- Metrics collection
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state machine."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class BaseConnector(ABC):
    """
    Abstract base class for all exchange connectors.
    
    Provides common functionality:
    - Callback management for message processing
    - Exponential backoff reconnection
    - Health status tracking
    - Message counting for metrics
    
    Subclasses must implement:
    - connect(): Establish WebSocket/REST connection
    - disconnect(): Clean shutdown
    """
    
    EXCHANGE_NAME: str = "base"  # Override in subclass
    
    def __init__(self, callback: Optional[Callable[[Dict], None]] = None):
        """
        Initialize base connector.
        
        Args:
            callback: Async function to call with each normalized message.
                     Signature: async def callback(message: Dict) -> None
        """
        self._callback = callback
        self._state = ConnectionState.DISCONNECTED
        
        # Reconnection settings
        self._reconnect_delay = 1.0  # Current delay (grows with backoff)
        self._reconnect_delay_max = 60.0  # Maximum delay
        self._reconnect_delay_multiplier = 2.0  # Exponential factor
        
        # Metrics
        self._message_count = 0
        self._error_count = 0
        self._last_message_time: Optional[float] = None
        self._connect_time: Optional[float] = None
    
    def set_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set or update the message callback."""
        self._callback = callback
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """True if currently connected."""
        return self._state == ConnectionState.CONNECTED
    
    @property
    def message_count(self) -> int:
        """Total messages processed."""
        return self._message_count
    
    @property
    def last_message_age(self) -> Optional[float]:
        """Seconds since last message, or None if no messages yet."""
        if self._last_message_time is None:
            return None
        return time.time() - self._last_message_time
    
    @property
    def is_stale(self) -> bool:
        """True if no messages received in last 60 seconds."""
        age = self.last_message_age
        return age is not None and age > 60.0
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get connector health status.
        
        Returns dict suitable for JSON health endpoint.
        """
        return {
            "exchange": self.EXCHANGE_NAME,
            "state": self._state.value,
            "is_connected": self.is_connected,
            "is_stale": self.is_stale,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_message_age_seconds": self.last_message_age,
            "uptime_seconds": (
                time.time() - self._connect_time 
                if self._connect_time else None
            ),
        }
    
    async def _safe_callback(self, message: Dict[str, Any]) -> None:
        """
        Call the callback with error handling.
        
        If callback fails, log error but don't crash the connector.
        """
        if not self._callback:
            return
        
        try:
            # Support both sync and async callbacks
            result = self._callback(message)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Callback error for {self.EXCHANGE_NAME}: {e}")
            self._error_count += 1
    
    async def _backoff(self) -> None:
        """
        Wait with exponential backoff before reconnecting.
        
        Delay starts at 1s, doubles each time, maxes at 60s.
        """
        logger.info(
            f"{self.EXCHANGE_NAME}: Waiting {self._reconnect_delay:.1f}s "
            f"before reconnecting..."
        )
        await asyncio.sleep(self._reconnect_delay)
        
        # Increase delay for next time (exponential backoff)
        self._reconnect_delay = min(
            self._reconnect_delay * self._reconnect_delay_multiplier,
            self._reconnect_delay_max
        )
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the exchange and start processing messages.
        
        This method should run indefinitely, handling reconnection internally.
        Override in subclass.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully disconnect from the exchange.
        
        Override in subclass.
        """
        raise NotImplementedError
