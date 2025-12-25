"""
Sentiment Data Producer for Kafka
===================================

Streams real-time sentiment from multiple sources to Kafka:
- CryptoPanic (news)
- Telegram (alpha channels)
- Multi-model ensemble analysis

Publishes to: sentiment_signals topic
Polling: 30-second intervals
Format: JSON with timestamp, source, symbol, score, signal

Usage:
    producer = SentimentProducer(
        cryptopanic_key="your_key",
        kafka_bootstrap="localhost:9092"
    )
    producer.start()
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from kafka import KafkaProducer
from kafka.errors import KafkaError

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors import (
    create_cryptopanic_connector,
    create_telegram_connector,
)

logger = logging.getLogger(__name__)


class SentimentProducer:
    """
    Real-time sentiment producer for Kafka.
    
    Polls multiple sentiment sources every 30s and publishes
    analyzed sentiment to Kafka for consumption by signal processor.
    
    Example:
        producer = SentimentProducer(
            cryptopanic_key="your_key",
            kafka_bootstrap="localhost:9092",
            poll_interval=30
        )
        
        # Run continuously
        producer.start()
        
        # Or single poll
        signals = producer.poll_all_sources()
    """
    
    def __init__(
        self,
        cryptopanic_key: Optional[str] = None,
        kafka_bootstrap: str = "localhost:9092",
        kafka_topic: str = "sentiment_signals",
        poll_interval: int = 30,
        symbols: List[str] = None,
        telegram_channels: List[str] = None,
    ):
        """
        Initialize sentiment producer.
        
        Args:
            cryptopanic_key: CryptoPanic API key
            kafka_bootstrap: Kafka bootstrap servers
            kafka_topic: Topic to publish to
            poll_interval: Seconds between polls
            symbols: Crypto symbols to track
            telegram_channels: Telegram channels to monitor
        """
        self.cryptopanic_key = cryptopanic_key or os.getenv("CRYPTOPANIC_API_KEY")
        self.kafka_bootstrap = kafka_bootstrap
        self.kafka_topic = kafka_topic
        self.poll_interval = poll_interval
        
        self.symbols = symbols or ["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "XRP"]
        self.telegram_channels = telegram_channels or [
            "@cryptowhalesignal",
            "@cryptoalphacalls",
        ]
        
        # Initialize connectors
        self.cryptopanic = create_cryptopanic_connector(
            api_key=self.cryptopanic_key,
            with_analyzer=True,
        )
        
        self.telegram = create_telegram_connector(with_analyzer=True)
        
        # Initialize Kafka producer
        self.producer = None
        self._init_kafka()
        
        # Stats
        self.messages_sent = 0
        self.errors = 0
        self.start_time = None
        
        logger.info(f"SentimentProducer initialized (poll interval: {poll_interval}s)")
    
    def _init_kafka(self):
        """Initialize Kafka producer."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,  # Preserve order
            )
            logger.info(f"✓ Connected to Kafka: {self.kafka_bootstrap}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None
    
    def poll_cryptopanic(self) -> List[Dict[str, Any]]:
        """Poll CryptoPanic for news sentiment."""
        signals = []
        
        for symbol in self.symbols:
            try:
                result = self.cryptopanic.analyze_sentiment(
                    currencies=symbol,
                    filter="hot"
                )
                
                if result.get('news_count', 0) > 0:
                    signals.append({
                        'source': 'cryptopanic',
                        'symbol': f"{symbol}USDT",
                        'timestamp': datetime.utcnow().isoformat(),
                        'score': result.get('sentiment_score', 0),
                        'signal': result.get('signal', 'NEUTRAL'),
                        'confidence': result.get('model_sentiment', 0),
                        'data_points': result.get('news_count', 0),
                        'metadata': {
                            'bullish_pct': result.get('bullish_pct', 0),
                            'bearish_pct': result.get('bearish_pct', 0),
                        }
                    })
                    
                    logger.debug(f"CryptoPanic {symbol}: {result.get('signal')} ({result.get('sentiment_score', 0):+.2f})")
            
            except Exception as e:
                logger.error(f"CryptoPanic poll failed for {symbol}: {e}")
                self.errors += 1
        
        return signals
    
    def poll_telegram(self) -> List[Dict[str, Any]]:
        """Poll Telegram channels for sentiment."""
        signals = []
        
        for channel in self.telegram_channels:
            try:
                result = self.telegram.analyze_sentiment(channel, limit=10)
                
                if result.get('message_count', 0) > 0:
                    # Extract primary symbol from messages (basic heuristic)
                    # In production, would use NER or keyword extraction
                    signals.append({
                        'source': 'telegram',
                        'channel': channel,
                        'symbol': 'BTCUSDT',  # Default to BTC for now
                        'timestamp': datetime.utcnow().isoformat(),
                        'score': result.get('sentiment_score', 0),
                        'signal': result.get('signal', 'NEUTRAL'),
                        'confidence': result.get('weighted_score', 0),
                        'data_points': result.get('message_count', 0),
                        'metadata': {
                            'total_views': result.get('total_views', 0),
                            'bullish_pct': result.get('bullish_pct', 0),
                            'bearish_pct': result.get('bearish_pct', 0),
                        }
                    })
                    
                    logger.debug(f"Telegram {channel}: {result.get('signal')} ({result.get('sentiment_score', 0):+.2f})")
            
            except Exception as e:
                logger.error(f"Telegram poll failed for {channel}: {e}")
                self.errors += 1
        
        return signals
    
    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate signals by symbol.
        
        Args:
            signals: List of individual signals
            
        Returns:
            Dict mapping symbols to aggregated sentiment
        """
        aggregated = {}
        
        for signal in signals:
            symbol = signal.get('symbol', 'BTCUSDT')
            
            if symbol not in aggregated:
                aggregated[symbol] = {
                    'symbol': symbol,
                    'timestamp': signal['timestamp'],
                    'sources': [],
                    'weighted_score': 0,
                    'total_weight': 0,
                    'signal': 'NEUTRAL',
                }
            
            # Weight by source and data points
            if signal['source'] == 'cryptopanic':
                weight = 0.6 * min(signal['data_points'], 5)  # Cap at 5 news items
            elif signal['source'] == 'telegram':
                weight = 0.4 * min(signal['data_points'] / 10, 1)  # Normalize by 10 messages
            else:
                weight = 0.5
            
            aggregated[symbol]['sources'].append({
                'source': signal['source'],
                'score': signal['score'],
                'signal': signal['signal'],
                'weight': weight,
            })
            
            aggregated[symbol]['weighted_score'] += signal['score'] * weight
            aggregated[symbol]['total_weight'] += weight
        
        # Calculate final scores
        for symbol, data in aggregated.items():
            if data['total_weight'] > 0:
                final_score = data['weighted_score'] / data['total_weight']
                data['sentiment_score'] = final_score
                
                if final_score > 0.3:
                    data['signal'] = 'BULLISH'
                elif final_score < -0.3:
                    data['signal'] = 'BEARISH'
                else:
                    data['signal'] = 'NEUTRAL'
        
        return aggregated
    
    def poll_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Poll all sentiment sources and aggregate."""
        all_signals = []
        
        # Poll CryptoPanic
        all_signals.extend(self.poll_cryptopanic())
        
        # Poll Telegram
        all_signals.extend(self.poll_telegram())
        
        # Aggregate by symbol
        aggregated = self.aggregate_signals(all_signals)
        
        return aggregated
    
    def publish_to_kafka(self, signals: Dict[str, Dict[str, Any]]):
        """Publish sentiment signals to Kafka."""
        if not self.producer:
            logger.warning("Kafka producer not initialized, skipping publish")
            return
        
        for symbol, signal_data in signals.items():
            try:
                future = self.producer.send(
                    self.kafka_topic,
                    key=symbol,
                    value=signal_data
                )
                
                # Wait for confirmation (with timeout)
                record_metadata = future.get(timeout=10)
                
                self.messages_sent += 1
                
                logger.info(
                    f"Published {symbol}: {signal_data['signal']} "
                    f"({signal_data['sentiment_score']:+.2f}) "
                    f"[partition {record_metadata.partition}, offset {record_metadata.offset}]"
                )
            
            except KafkaError as e:
                logger.error(f"Failed to publish {symbol}: {e}")
                self.errors += 1
    
    def start(self):
        """Start continuous polling and publishing."""
        self.start_time = time.time()
        
        logger.info(f"Starting sentiment producer (poll interval: {self.poll_interval}s)")
        logger.info(f"Tracking symbols: {', '.join(self.symbols)}")
        logger.info(f"Monitoring channels: {', '.join(self.telegram_channels)}")
        logger.info(f"Publishing to: {self.kafka_topic}")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Poll all sources
                signals = self.poll_all_sources()
                
                # Publish to Kafka
                if signals:
                    self.publish_to_kafka(signals)
                else:
                    logger.warning("No signals generated this cycle")
                
                # Log stats
                runtime = time.time() - self.start_time
                logger.info(
                    f"Stats: {self.messages_sent} sent, {self.errors} errors, "
                    f"{runtime:.0f}s runtime"
                )
                
                # Sleep until next poll
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.poll_interval - cycle_time)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping {sleep_time:.1f}s until next poll")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Shutting down sentiment producer...")
            if self.producer:
                self.producer.flush()
                self.producer.close()
            logger.info("Shutdown complete")
    
    def health_check(self) -> Dict[str, Any]:
        """Check producer health."""
        return {
            'status': 'healthy' if self.producer else 'degraded',
            'kafka_connected': self.producer is not None,
            'messages_sent': self.messages_sent,
            'errors': self.errors,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'symbols_tracked': len(self.symbols),
            'channels_monitored': len(self.telegram_channels),
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Sentiment Data Producer for Kafka")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="sentiment_signals", help="Kafka topic")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval (seconds)")
    parser.add_argument("--test", action="store_true", help="Run single poll test")
    
    args = parser.parse_args()
    
    producer = SentimentProducer(
        kafka_bootstrap=args.kafka,
        kafka_topic=args.topic,
        poll_interval=args.interval,
    )
    
    if args.test:
        print("=" * 60)
        print("SENTIMENT PRODUCER TEST")
        print("=" * 60)
        
        print("\nPolling all sources...")
        signals = producer.poll_all_sources()
        
        print(f"\nGenerated {len(signals)} signals:")
        for symbol, data in signals.items():
            print(f"\n{symbol}:")
            print(f"  Signal: {data['signal']}")
            print(f"  Score: {data['sentiment_score']:+.2f}")
            print(f"  Sources: {len(data['sources'])}")
        
        print(f"\nHealth: {producer.health_check()}")
    else:
        producer.start()
