"""
Data Interface Layer

Connects Layer 1 Explorer to the broader HIMARI ecosystem:
- Redis: Feature store (Layer 0 features, strategy cache)
- Kafka: Event bus (regime signals, strategy candidates, feedback)
- Neo4j: Knowledge graph (strategy relationships, market structure)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Layer1Config:
    """Configuration for Layer 1 data connections."""
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    feature_key_prefix: str = "layer0:features:"
    strategy_cache_prefix: str = "layer1:strategies:"
    feature_ttl_seconds: int = 300  # 5 minutes

    # Kafka configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "layer1-explorer"
    regime_topic: str = "himari.regime.signals"
    strategy_topic: str = "himari.strategy.candidates"
    feedback_topic: str = "himari.strategy.feedback"

    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Connection settings
    connection_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class RegimeSignal:
    """Regime signal from Layer 4."""
    regime_id: int
    regime_name: str
    confidence: float
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RegimeSignal':
        return cls(
            regime_id=data['regime_id'],
            regime_name=data['regime_name'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            features=data.get('features', {})
        )


@dataclass
class StrategyFeedback:
    """Feedback on strategy performance from execution layer."""
    strategy_id: str
    live_sharpe: float
    live_drawdown: float
    transfer_ratio: float
    trade_count: int
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyFeedback':
        return cls(
            strategy_id=data['strategy_id'],
            live_sharpe=data['live_sharpe'],
            live_drawdown=data['live_drawdown'],
            transfer_ratio=data['transfer_ratio'],
            trade_count=data['trade_count'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            details=data.get('details', {})
        )


class RedisFeatureStore:
    """
    Redis client for feature store access.

    Reads 60-dim feature vectors from Layer 0.
    Caches strategy genomes for quick retrieval.
    """

    def __init__(self, config: Layer1Config):
        self.config = config
        self._client = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            import redis.asyncio as redis

            self._client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_timeout=self.config.connection_timeout
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            return True

        except ImportError:
            logger.warning("redis package not installed, using mock client")
            self._client = MockRedisClient()
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = MockRedisClient()
            self._connected = True
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
        self._connected = False

    async def get_features(self, symbol: str = "BTCUSDT") -> Optional[np.ndarray]:
        """
        Get latest 60-dim feature vector for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            60-dim numpy array or None if not available
        """
        if not self._connected:
            await self.connect()

        try:
            key = f"{self.config.feature_key_prefix}{symbol}"
            data = await self._client.get(key)

            if data:
                features = json.loads(data)
                return np.array(features['vector'])
            return None

        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None

    async def get_feature_history(
        self,
        symbol: str = "BTCUSDT",
        lookback_minutes: int = 60
    ) -> List[np.ndarray]:
        """Get historical feature vectors."""
        if not self._connected:
            await self.connect()

        try:
            key = f"{self.config.feature_key_prefix}{symbol}:history"
            data = await self._client.lrange(key, 0, lookback_minutes - 1)

            return [np.array(json.loads(d)['vector']) for d in data]

        except Exception as e:
            logger.error(f"Error getting feature history: {e}")
            return []

    async def cache_strategy(
        self,
        strategy_id: str,
        strategy_data: Dict,
        ttl_seconds: int = 3600
    ) -> bool:
        """Cache a strategy genome."""
        if not self._connected:
            await self.connect()

        try:
            key = f"{self.config.strategy_cache_prefix}{strategy_id}"
            await self._client.setex(
                key,
                ttl_seconds,
                json.dumps(strategy_data)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching strategy: {e}")
            return False

    async def get_cached_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Retrieve a cached strategy."""
        if not self._connected:
            await self.connect()

        try:
            key = f"{self.config.strategy_cache_prefix}{strategy_id}"
            data = await self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached strategy: {e}")
            return None


class KafkaEventBus:
    """
    Kafka client for event-driven communication.

    Consumes:
    - Regime signals from Layer 4
    - Strategy feedback from execution layer

    Produces:
    - Strategy candidates to Layer 2
    """

    def __init__(self, config: Layer1Config):
        self.config = config
        self._producer = None
        self._consumer = None
        self._connected = False
        self._regime_handlers: List[Callable] = []
        self._feedback_handlers: List[Callable] = []

    async def connect(self) -> bool:
        """Establish Kafka connections."""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            # Create producer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self._producer.start()

            # Create consumer
            self._consumer = AIOKafkaConsumer(
                self.config.regime_topic,
                self.config.feedback_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest'
            )
            await self._consumer.start()

            self._connected = True
            logger.info(f"Connected to Kafka at {self.config.kafka_bootstrap_servers}")
            return True

        except ImportError:
            logger.warning("aiokafka package not installed, using mock client")
            self._producer = MockKafkaProducer()
            self._consumer = MockKafkaConsumer()
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._producer = MockKafkaProducer()
            self._consumer = MockKafkaConsumer()
            self._connected = True
            return False

    async def disconnect(self) -> None:
        """Close Kafka connections."""
        if self._producer and hasattr(self._producer, 'stop'):
            await self._producer.stop()
        if self._consumer and hasattr(self._consumer, 'stop'):
            await self._consumer.stop()
        self._connected = False

    def on_regime_signal(self, handler: Callable[[RegimeSignal], None]) -> None:
        """Register handler for regime signals."""
        self._regime_handlers.append(handler)

    def on_feedback(self, handler: Callable[[StrategyFeedback], None]) -> None:
        """Register handler for strategy feedback."""
        self._feedback_handlers.append(handler)

    async def start_consuming(self) -> None:
        """Start consuming messages (blocking)."""
        if not self._connected:
            await self.connect()

        try:
            async for msg in self._consumer:
                if msg.topic == self.config.regime_topic:
                    signal = RegimeSignal.from_dict(msg.value)
                    for handler in self._regime_handlers:
                        await self._safe_call(handler, signal)

                elif msg.topic == self.config.feedback_topic:
                    feedback = StrategyFeedback.from_dict(msg.value)
                    for handler in self._feedback_handlers:
                        await self._safe_call(handler, feedback)

        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")

    async def _safe_call(self, handler: Callable, data: Any) -> None:
        """Safely call handler with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"Handler error: {e}")

    async def publish_strategy_candidate(
        self,
        strategy_data: Dict
    ) -> bool:
        """Publish a validated strategy candidate to Layer 2."""
        if not self._connected:
            await self.connect()

        try:
            await self._producer.send_and_wait(
                self.config.strategy_topic,
                strategy_data
            )
            logger.debug(f"Published strategy {strategy_data.get('id', 'unknown')[:8]}")
            return True
        except Exception as e:
            logger.error(f"Error publishing strategy: {e}")
            return False

    async def publish_batch(
        self,
        strategies: List[Dict]
    ) -> int:
        """Publish multiple strategy candidates."""
        if not self._connected:
            await self.connect()

        success_count = 0
        for strategy in strategies:
            if await self.publish_strategy_candidate(strategy):
                success_count += 1

        return success_count


class Neo4jKnowledgeGraph:
    """
    Neo4j client for knowledge graph queries.

    Queries:
    - Strategy relationships and lineage
    - Market structure knowledge
    - Feature correlations
    """

    def __init__(self, config: Layer1Config):
        self.config = config
        self._driver = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish Neo4j connection."""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )

            # Test connection
            async with self._driver.session() as session:
                await session.run("RETURN 1")

            self._connected = True
            logger.info(f"Connected to Neo4j at {self.config.neo4j_uri}")
            return True

        except ImportError:
            logger.warning("neo4j package not installed, using mock client")
            self._driver = MockNeo4jDriver()
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = MockNeo4jDriver()
            self._connected = True
            return False

    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver and hasattr(self._driver, 'close'):
            await self._driver.close()
        self._connected = False

    async def store_strategy(
        self,
        strategy_id: str,
        genome_vector: List[float],
        metrics: Dict[str, float],
        parent_ids: List[str] = None
    ) -> bool:
        """Store a strategy in the knowledge graph."""
        if not self._connected:
            await self.connect()

        try:
            query = """
            MERGE (s:Strategy {id: $strategy_id})
            SET s.genome_vector = $genome_vector,
                s.sharpe = $sharpe,
                s.max_drawdown = $max_drawdown,
                s.created_at = datetime()
            """

            async with self._driver.session() as session:
                await session.run(
                    query,
                    strategy_id=strategy_id,
                    genome_vector=genome_vector,
                    sharpe=metrics.get('sharpe', 0),
                    max_drawdown=metrics.get('max_drawdown', 0)
                )

                # Create parent relationships
                if parent_ids:
                    for parent_id in parent_ids:
                        await session.run(
                            """
                            MATCH (child:Strategy {id: $child_id})
                            MATCH (parent:Strategy {id: $parent_id})
                            MERGE (child)-[:DERIVED_FROM]->(parent)
                            """,
                            child_id=strategy_id,
                            parent_id=parent_id
                        )

            return True

        except Exception as e:
            logger.error(f"Error storing strategy in graph: {e}")
            return False

    async def find_similar_strategies(
        self,
        genome_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Find strategies similar to the given genome vector."""
        if not self._connected:
            await self.connect()

        try:
            # Use cosine similarity via Neo4j GDS if available
            query = """
            MATCH (s:Strategy)
            WHERE s.genome_vector IS NOT NULL
            WITH s, gds.similarity.cosine(s.genome_vector, $query_vector) AS similarity
            WHERE similarity >= $min_similarity
            RETURN s.id AS id, s.sharpe AS sharpe, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    query_vector=genome_vector,
                    min_similarity=min_similarity,
                    limit=limit
                )
                return [dict(record) async for record in result]

        except Exception as e:
            logger.error(f"Error finding similar strategies: {e}")
            return []

    async def get_strategy_lineage(
        self,
        strategy_id: str,
        depth: int = 3
    ) -> List[Dict]:
        """Get the lineage (ancestry) of a strategy."""
        if not self._connected:
            await self.connect()

        try:
            query = """
            MATCH path = (s:Strategy {id: $strategy_id})-[:DERIVED_FROM*1..$depth]->(ancestor:Strategy)
            RETURN ancestor.id AS id,
                   ancestor.sharpe AS sharpe,
                   length(path) AS generation
            ORDER BY generation
            """

            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    strategy_id=strategy_id,
                    depth=depth
                )
                return [dict(record) async for record in result]

        except Exception as e:
            logger.error(f"Error getting strategy lineage: {e}")
            return []

    async def get_top_strategies(
        self,
        regime: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get top performing strategies, optionally filtered by regime."""
        if not self._connected:
            await self.connect()

        try:
            if regime:
                query = """
                MATCH (s:Strategy)-[:PERFORMS_WELL_IN]->(r:Regime {name: $regime})
                RETURN s.id AS id, s.sharpe AS sharpe, s.max_drawdown AS max_drawdown
                ORDER BY s.sharpe DESC
                LIMIT $limit
                """
                params = {'regime': regime, 'limit': limit}
            else:
                query = """
                MATCH (s:Strategy)
                RETURN s.id AS id, s.sharpe AS sharpe, s.max_drawdown AS max_drawdown
                ORDER BY s.sharpe DESC
                LIMIT $limit
                """
                params = {'limit': limit}

            async with self._driver.session() as session:
                result = await session.run(query, **params)
                return [dict(record) async for record in result]

        except Exception as e:
            logger.error(f"Error getting top strategies: {e}")
            return []


class Layer1DataInterface:
    """
    Unified data interface for Layer 1 Explorer.

    Coordinates access to all external data stores:
    - Redis (features)
    - Kafka (events)
    - Neo4j (knowledge graph)
    """

    def __init__(self, config: Optional[Layer1Config] = None):
        self.config = config or Layer1Config()

        self.redis = RedisFeatureStore(self.config)
        self.kafka = KafkaEventBus(self.config)
        self.neo4j = Neo4jKnowledgeGraph(self.config)

        self._connected = False

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all data stores."""
        results = {}

        results['redis'] = await self.redis.connect()
        results['kafka'] = await self.kafka.connect()
        results['neo4j'] = await self.neo4j.connect()

        self._connected = all(results.values())

        logger.info(f"Data interface connection results: {results}")
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all data stores."""
        await self.redis.disconnect()
        await self.kafka.disconnect()
        await self.neo4j.disconnect()
        self._connected = False

    async def get_current_context(
        self,
        symbol: str = "BTCUSDT"
    ) -> Dict[str, Any]:
        """
        Get current market context for strategy generation.

        Returns:
            Dict containing features, regime info, and top strategies
        """
        context = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'features': None,
            'regime': None,
            'top_strategies': []
        }

        # Get features
        features = await self.redis.get_features(symbol)
        if features is not None:
            context['features'] = features.tolist()

        # Get top strategies for context
        top_strategies = await self.neo4j.get_top_strategies(limit=10)
        context['top_strategies'] = top_strategies

        return context

    async def publish_validated_strategies(
        self,
        strategies: List[Dict]
    ) -> int:
        """Publish validated strategies to downstream layers."""
        # Store in knowledge graph
        for strategy in strategies:
            await self.neo4j.store_strategy(
                strategy_id=strategy['id'],
                genome_vector=strategy.get('vector', []),
                metrics=strategy.get('metrics', {}),
                parent_ids=strategy.get('parent_ids', [])
            )

        # Publish to Kafka
        published = await self.kafka.publish_batch(strategies)

        return published


# Mock implementations for testing without infrastructure

class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}

    async def ping(self):
        return True

    async def get(self, key: str):
        return self._data.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._data[key] = value

    async def lrange(self, key: str, start: int, end: int):
        data = self._data.get(key, [])
        return data[start:end+1]

    async def close(self):
        pass


class MockKafkaProducer:
    """Mock Kafka producer for testing."""

    def __init__(self):
        self._messages = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_and_wait(self, topic: str, value: Any):
        self._messages.append({'topic': topic, 'value': value})


class MockKafkaConsumer:
    """Mock Kafka consumer for testing."""

    def __init__(self):
        self._messages = []

    async def start(self):
        pass

    async def stop(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class MockNeo4jDriver:
    """Mock Neo4j driver for testing."""

    def __init__(self):
        self._strategies = {}

    def session(self):
        return MockNeo4jSession(self._strategies)

    async def close(self):
        pass


class MockNeo4jSession:
    """Mock Neo4j session."""

    def __init__(self, strategies: Dict):
        self._strategies = strategies

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query: str, **kwargs):
        return MockNeo4jResult([])


class MockNeo4jResult:
    """Mock Neo4j result."""

    def __init__(self, records: List):
        self._records = records

    def __aiter__(self):
        return iter(self._records).__aiter__()
