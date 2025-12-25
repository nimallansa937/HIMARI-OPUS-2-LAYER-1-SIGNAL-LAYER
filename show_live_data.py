"""Show live data from Kafka topic."""
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'raw_market_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    consumer_timeout_ms=5000
)

print('=== LIVE DATA FROM KAFKA ===')
print()

count = 0
for msg in consumer:
    data = msg.value
    msg_type = data.get('type', 'unknown')
    symbol = data.get('symbol', '?')
    
    if msg_type == 'ohlcv':
        print(f"[OHLCV] {symbol}: O={data['open']:.2f} H={data['high']:.2f} L={data['low']:.2f} C={data['close']:.2f}")
    elif msg_type == 'orderbook':
        print(f"[L2] {symbol}: Bid={data['best_bid']:.2f} Ask={data['best_ask']:.2f} OBI={data['order_book_imbalance']:+.3f}")
    elif msg_type == 'trade':
        side = 'SELL' if data.get('is_buyer_maker') else 'BUY'
        print(f"[TRADE] {symbol}: {side} {data['quantity']:.4f} @ ${data['price']:,.2f}")
    else:
        print(f"[{msg_type.upper()}] {symbol}")
    
    count += 1
    if count >= 30:
        break

print()
print(f"Showed {count} messages")
