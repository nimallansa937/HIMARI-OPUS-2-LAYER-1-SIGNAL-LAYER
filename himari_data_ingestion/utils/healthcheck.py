"""
Health Check Utilities

Monitors connection status of all data sources and Kafka.
Provides health endpoints for container orchestration (K8s, Docker).
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    last_check: float
    message_count: int = 0
    last_message_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def is_stale(self) -> bool:
        """True if no messages received in 60 seconds."""
        if self.last_message_time is None:
            return True
        return time.time() - self.last_message_time > 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check,
            "message_count": self.message_count,
            "last_message_time": self.last_message_time,
            "is_stale": self.is_stale,
            "error": self.error,
        }


class HealthChecker:
    """
    Aggregates health status from all ingestion components.
    
    Usage:
        checker = HealthChecker()
        checker.register("binance", binance_connector)
        checker.register("kafka", kafka_publisher)
        
        health = checker.get_health()
        if health["status"] == "unhealthy":
            alert_ops_team()
    """
    
    def __init__(self, stale_threshold: float = 60.0):
        """
        Initialize health checker.
        
        Args:
            stale_threshold: Seconds without data to consider source stale
        """
        self.stale_threshold = stale_threshold
        self._components: Dict[str, Any] = {}
    
    def register(self, name: str, component: Any) -> None:
        """
        Register a component to monitor.
        
        Component must have a get_health() method that returns:
        {
            "is_connected": bool,
            "message_count": int,
            "state": str,
            ...
        }
        """
        self._components[name] = component
    
    def unregister(self, name: str) -> None:
        """Remove a component from monitoring."""
        self._components.pop(name, None)
    
    def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component."""
        component = self._components.get(name)
        
        if component is None:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                error="Component not found",
            )
        
        try:
            health = component.get_health()
            
            # Determine status
            if health.get("is_connected"):
                if health.get("is_stale"):
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.UNHEALTHY
            
            return ComponentHealth(
                name=name,
                status=status,
                last_check=time.time(),
                message_count=health.get("message_count", 0),
                last_message_time=health.get("last_message_time"),
            )
            
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                error=str(e),
            )
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get aggregated health status of all components.
        
        Returns: {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": float,
            "components": {
                "binance": {...},
                "kraken": {...},
                ...
            },
            "summary": {
                "total": int,
                "healthy": int,
                "degraded": int,
                "unhealthy": int,
            }
        }
        """
        component_health = {}
        summary = {"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0}
        
        for name in self._components:
            health = self.check_component(name)
            component_health[name] = health.to_dict()
            
            summary["total"] += 1
            if health.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif health.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            else:
                summary["unhealthy"] += 1
        
        # Determine overall status
        if summary["unhealthy"] > 0:
            overall = HealthStatus.UNHEALTHY
        elif summary["degraded"] > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "components": component_health,
            "summary": summary,
        }
    
    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        health = self.get_health()
        return health["status"] != "unhealthy"
