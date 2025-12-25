"""
Kafka Publisher for HIMARI Data Ingestion

This is the bridge between exchange connectors and your Redpanda cluster.
It takes normalized OHLCV messages and publishes them to the 'raw_market_data'
topic, where your Flink pipeline picks them up for quality validation.

Key features:
- Async publishing with batching for throughput
- Automatic retry with backoff on failures
- Local buffer when Kafka is temporarily unavailable
- Metrics for monitoring publish rates and latency
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import confluent-kafka (faster) or fall back to kafka-python
try:
    from confluent_kafka import Producer, KafkaError
    USING_CONFLUENT = True
    logger.info("Using confluent-kafka (recommended)")
except ImportError:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    USING_CONFLUENT = False
    logger.info("Using kafka-python (confluent-kafka recommended for better performance)")


@dataclass
class PublisherMetrics:
    """Track publisher performance."""
    messages_sent: int = 0
    messages_failed: int = 0
    messages_buffered: int = 0
    bytes_sent: int = 0
    last_send_time: Optional[float] = None
    last_error: Optional[str] = None
    avg_latency_ms: float = 0.0
    
    # Running latency calculation
    _latency_sum: float = 0.0
    _latency_count: int = 0


class KafkaPublisher:
    """
    Publishes normalized market data to Kafka/Redpanda.
    
    Usage:
        publisher = KafkaPublisher(
            bootstrap_servers="localhost:9092",
            topic="raw_market_data"
        )
        await publisher.connect()
        
        # From your connector callback:
        await publisher.publish(message)
        
        # On shutdown:
        await publisher.disconnect()
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "raw_market_data",
        buffer_size: int = 10000,
        batch_size: int = 100,
        linger_ms: int = 5,
    ):
        """
        Initialize Kafka publisher.
        
        Args:
            bootstrap_servers: Kafka/Redpanda broker addresses
            topic: Topic to publish to
            buffer_size: Max messages to buffer when Kafka unavailable
            batch_size: Messages to batch before sending
            linger_ms: Max time to wait for batch to fill
        """
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._linger_ms = linger_ms
        
        self._producer: Optional[Any] = None
        self._buffer: deque = deque(maxlen=buffer_size)
        self._metrics = PublisherMetrics()
        self._connected = False
        
        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """
        Connect to Kafka/Redpanda.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if USING_CONFLUENT:
                # Confluent Kafka configuration
                self._producer = Producer({
                    'bootstrap.servers': self._bootstrap_servers,
                    'client.id': 'himari-ingestion',
                    'acks': 'all',
                    'retries': 3,
                    'retry.backoff.ms': 100,
                    'linger.ms': self._linger_ms,
                    'batch.size': 16384,
                    'compression.type': 'none',
                    # Error callback
                    'error_cb': self._on_error,
                    # Delivery callback
                    'on_delivery': self._on_delivery,
                })
            else:
                # kafka-python configuration
                self._producer = KafkaProducer(
                    bootstrap_servers=self._bootstrap_servers.split(','),
                    client_id='himari-ingestion',
                    acks='all',
                    retries=3,
                    linger_ms=self._linger_ms,
                    batch_size=16384,
                    compression_type='gzip',
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                )
            
            self._connected = True
            logger.info(f"Connected to Kafka: {self._bootstrap_servers}")
            
            # Start background flush task
            self._flush_task = asyncio.create_task(self._background_flush())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._metrics.last_error = str(e)
            return False
    
    async def publish(self, message: Dict[str, Any]) -> bool:
        """
        Publish a message to Kafka.
        
        If Kafka is unavailable, the message is buffered locally.
        
        Args:
            message: Normalized OHLCV message
            
        Returns:
            True if sent (or buffered), False if buffer full
        """
        if not self._connected or not self._producer:
            # Buffer message for later
            if len(self._buffer) < self._buffer_size:
                self._buffer.append(message)
                self._metrics.messages_buffered = len(self._buffer)
                return True
            else:
                logger.warning("Buffer full, dropping message")
                self._metrics.messages_failed += 1
                return False
        
        try:
            # Use symbol as partition key for ordering
            key = message.get("symbol", "unknown")
            
            # Serialize
            value = json.dumps(message).encode('utf-8')
            
            # Record send time for latency tracking
            send_time = time.time()
            
            if USING_CONFLUENT:
                # Confluent Kafka - async produce
                self._producer.produce(
                    topic=self._topic,
                    key=key.encode('utf-8'),
                    value=value,
                    timestamp=int(send_time * 1000),
                )
                # Trigger delivery (non-blocking poll)
                self._producer.poll(0)
            else:
                # kafka-python - already serialized via value_serializer
                self._producer.send(
                    topic=self._topic,
                    key=key,
                    value=message,  # Serialized by producer
                    timestamp_ms=int(send_time * 1000),
                )
            
            # Update metrics
            self._metrics.messages_sent += 1
            self._metrics.bytes_sent += len(value)
            self._metrics.last_send_time = send_time
            
            # Track latency
            latency_ms = (time.time() - send_time) * 1000
            self._update_latency(latency_ms)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self._metrics.messages_failed += 1
            self._metrics.last_error = str(e)
            
            # Buffer for retry
            if len(self._buffer) < self._buffer_size:
                self._buffer.append(message)
            
            return False
    
    async def publish_batch(self, messages: List[Dict[str, Any]]) -> int:
        """
        Publish multiple messages efficiently.
        
        Args:
            messages: List of normalized OHLCV messages
            
        Returns:
            Number of messages successfully sent/buffered
        """
        sent = 0
        for msg in messages:
            if await self.publish(msg):
                sent += 1
        return sent
    
    async def _background_flush(self) -> None:
        """
        Background task to flush buffered messages.
        
        Runs every second, attempting to send any buffered messages.
        """
        while self._connected:
            await asyncio.sleep(1.0)
            
            # Try to send buffered messages
            while self._buffer and self._connected:
                message = self._buffer.popleft()
                if not await self.publish(message):
                    # Failed again, put back at front
                    self._buffer.appendleft(message)
                    break
            
            # Flush producer
            if self._producer:
                if USING_CONFLUENT:
                    self._producer.poll(0)
                else:
                    self._producer.flush(timeout=0.1)
            
            self._metrics.messages_buffered = len(self._buffer)
    
    def _update_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        self._metrics._latency_sum += latency_ms
        self._metrics._latency_count += 1
        self._metrics.avg_latency_ms = (
            self._metrics._latency_sum / self._metrics._latency_count
        )
    
    def _on_error(self, err: Any) -> None:
        """Confluent Kafka error callback."""
        logger.error(f"Kafka error: {err}")
        self._metrics.last_error = str(err)
    
    def _on_delivery(self, err: Any, msg: Any) -> None:
        """Confluent Kafka delivery callback."""
        if err:
            logger.warning(f"Message delivery failed: {err}")
            self._metrics.messages_failed += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get publisher metrics for monitoring."""
        return {
            "connected": self._connected,
            "messages_sent": self._metrics.messages_sent,
            "messages_failed": self._metrics.messages_failed,
            "messages_buffered": self._metrics.messages_buffered,
            "bytes_sent": self._metrics.bytes_sent,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "last_error": self._metrics.last_error,
        }
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from Kafka."""
        self._connected = False
        
        # Cancel background task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining messages
        if self._producer:
            if USING_CONFLUENT:
                self._producer.flush(timeout=10)
            else:
                self._producer.flush(timeout=10)
                self._producer.close(timeout=5)
        
        logger.info("Disconnected from Kafka")
