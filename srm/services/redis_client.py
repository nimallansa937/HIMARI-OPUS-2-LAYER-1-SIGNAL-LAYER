"""
SRM Redis Client

Redis interface for SRM metric publication and retrieval.
Provides atomic operations for publishing risk scores and
fast reads for the main trading loop.
"""

import json
from typing import Optional
from datetime import datetime
import logging

try:
    import redis
except ImportError:
    redis = None

from ..composite import CompositeRiskResult

logger = logging.getLogger(__name__)


# Redis key patterns
RISK_CURRENT_KEY = "srm:risk:{symbol}"  # Current snapshot (hash)
RISK_HISTORY_KEY = "srm:history:{symbol}"  # Time series (sorted set)
RISK_REGIME_KEY = "srm:regime:{symbol}"  # Current regime (string)
RISK_ALERTS_KEY = "srm:alerts:{symbol}"  # Recent alerts (list)

# TTL settings
CURRENT_SNAPSHOT_TTL = 30  # Seconds - invalidate stale data
HISTORY_RETENTION = 86400 * 7  # 7 days of history


class SRMRedisClient:
    """
    Redis interface for SRM metric publication and retrieval.
    
    Provides atomic operations for publishing risk scores and
    fast reads for the main trading loop.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if redis is None:
            raise ImportError("redis package not installed. Run: pip install redis")
        
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)
        self._connected = False
        
        try:
            self.client.ping()
            self._connected = True
            logger.info(f"Redis connected: {redis_url}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}")
    
    @property
    def connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected:
            try:
                self.client.ping()
                self._connected = True
            except:
                self._connected = False
        return self._connected
    
    def publish_risk(self, symbol: str, result: CompositeRiskResult) -> bool:
        """
        Publish current risk metrics to Redis.
        
        Uses pipeline for atomic multi-key update.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            result: CompositeRiskResult from calculator
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Redis not connected, skipping publish")
            return False
        
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Prepare data
            current_data = {
                'score': str(result.score),
                'regime': result.regime.value,
                'fsi': str(result.signal_values.get('fsi', 0)),
                'lei': str(result.signal_values.get('lei', 0)),
                'ods': str(result.signal_values.get('ods', 0)),
                'scsi': str(result.signal_values.get('scsi', 0)),
                'lci': str(result.signal_values.get('lci', 0)),
                'caci': str(result.signal_values.get('caci', 0)),
                'amplified': str(result.amplification_applied),
                'risk_level': result.metadata.get('risk_level', 'UNKNOWN'),
                'timestamp': timestamp
            }
            
            # Atomic pipeline
            pipe = self.client.pipeline()
            
            # Current snapshot
            current_key = RISK_CURRENT_KEY.format(symbol=symbol)
            pipe.hset(current_key, mapping=current_data)
            pipe.expire(current_key, CURRENT_SNAPSHOT_TTL)
            
            # History (sorted set by timestamp)
            history_key = RISK_HISTORY_KEY.format(symbol=symbol)
            history_entry = json.dumps({**current_data, 'timestamp': timestamp})
            pipe.zadd(history_key, {history_entry: datetime.utcnow().timestamp()})
            
            # Trim history to retention window
            cutoff = datetime.utcnow().timestamp() - HISTORY_RETENTION
            pipe.zremrangebyscore(history_key, '-inf', cutoff)
            
            # Current regime
            regime_key = RISK_REGIME_KEY.format(symbol=symbol)
            pipe.set(regime_key, result.regime.value)
            
            pipe.execute()
            
            logger.debug(f"Published risk for {symbol}: score={result.score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish risk to Redis: {e}")
            return False
    
    def get_current_risk(self, symbol: str) -> Optional[dict]:
        """
        Read current risk snapshot for trading loop.
        
        Returns None if data is stale or missing.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dict with risk data or None
        """
        if not self.connected:
            return None
        
        try:
            key = RISK_CURRENT_KEY.format(symbol=symbol)
            data = self.client.hgetall(key)
            
            if not data:
                return None
            
            return {
                'score': float(data['score']),
                'regime': data['regime'],
                'fsi': float(data['fsi']),
                'lei': float(data['lei']),
                'ods': float(data['ods']),
                'scsi': float(data['scsi']),
                'lci': float(data['lci']),
                'caci': float(data['caci']),
                'amplified': data['amplified'] == 'True',
                'risk_level': data.get('risk_level', 'UNKNOWN'),
                'timestamp': data['timestamp']
            }
        except Exception as e:
            logger.error(f"Failed to get current risk: {e}")
            return None
    
    def get_risk_velocity(self, symbol: str, window_seconds: int = 300) -> Optional[float]:
        """
        Calculate rate of change in risk score over window.
        
        Used to detect rapid risk escalation. Returns None if insufficient
        data or if the time window is too small for meaningful calculation.
        
        Args:
            symbol: Trading pair symbol
            window_seconds: Lookback window in seconds (default: 5 minutes)
        
        Returns:
            Score change per second, or None if insufficient data or error
        """
        if not self.connected:
            return None
        
        # Minimum time delta for meaningful velocity calculation (5 seconds)
        MIN_TIME_DELTA: float = 5.0
        
        try:
            history_key: str = RISK_HISTORY_KEY.format(symbol=symbol)
            now: float = datetime.utcnow().timestamp()
            
            entries = self.client.zrangebyscore(
                history_key, 
                now - window_seconds, 
                now,
                withscores=True
            )
            
            if len(entries) < 2:
                return None
            
            # Parse scores with error handling for malformed entries
            scores: list = []
            for entry, ts in entries:
                try:
                    parsed = json.loads(entry)
                    score_val = parsed.get('score')
                    if score_val is not None:
                        scores.append((float(score_val), float(ts)))
                except (json.JSONDecodeError, ValueError, TypeError) as parse_err:
                    logger.debug(f"Skipping malformed history entry: {parse_err}")
                    continue
            
            if len(scores) < 2:
                return None
            
            scores.sort(key=lambda x: x[1])
            
            first_score: float = scores[0][0]
            first_ts: float = scores[0][1]
            last_score: float = scores[-1][0]
            last_ts: float = scores[-1][1]
            
            time_delta: float = last_ts - first_ts
            
            # Require minimum time delta for meaningful velocity calculation
            if time_delta < MIN_TIME_DELTA:
                return None
            
            velocity: float = (last_score - first_score) / time_delta
            return velocity
            
        except Exception as e:
            logger.error(f"Failed to calculate risk velocity: {e}")
            return None
    
    def get_risk_history(
        self, 
        symbol: str, 
        hours: int = 24
    ) -> list:
        """
        Get historical risk scores.
        
        Args:
            symbol: Trading pair symbol
            hours: Hours of history to retrieve
        
        Returns:
            List of (timestamp, score) tuples
        """
        if not self.connected:
            return []
        
        try:
            history_key = RISK_HISTORY_KEY.format(symbol=symbol)
            now = datetime.utcnow().timestamp()
            cutoff = now - (hours * 3600)
            
            entries = self.client.zrangebyscore(
                history_key, 
                cutoff, 
                now,
                withscores=True
            )
            
            return [
                (ts, float(json.loads(e)['score'])) 
                for e, ts in entries
            ]
            
        except Exception as e:
            logger.error(f"Failed to get risk history: {e}")
            return []
    
    def publish_alert(self, symbol: str, alert: dict) -> bool:
        """
        Push alert to alerts list.
        
        Args:
            symbol: Trading pair symbol
            alert: Alert data dict
        
        Returns:
            True if successful
        """
        if not self.connected:
            return False
        
        try:
            key = RISK_ALERTS_KEY.format(symbol=symbol)
            alert['timestamp'] = datetime.utcnow().isoformat()
            self.client.lpush(key, json.dumps(alert))
            self.client.ltrim(key, 0, 99)  # Keep last 100 alerts
            return True
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False
    
    def get_recent_alerts(self, symbol: str, count: int = 10) -> list:
        """
        Get recent alerts.
        
        Args:
            symbol: Trading pair symbol
            count: Number of alerts to retrieve
        
        Returns:
            List of alert dicts
        """
        if not self.connected:
            return []
        
        try:
            key = RISK_ALERTS_KEY.format(symbol=symbol)
            entries = self.client.lrange(key, 0, count - 1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.client.close()
            self._connected = False
        except:
            pass
