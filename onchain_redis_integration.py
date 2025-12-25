"""
On-Chain Redis Integration Module
==================================

Methods to add to SignalProcessor for on-chain signal publishing.

Publishes whale tracking and network health signals to Redis for
downstream consumption and monitoring.
"""

# Add these methods to the SignalProcessor class


def _init_onchain_redis(self):
    """
    Initialize on-chain Redis keys and primitives.
    
    Called from SignalProcessor.__init__
    """
    from primitives import (
        OnChainWhaleTracker, 
        OnChainNetworkHealth, 
        EnhancedCascadeDetector,
        is_onchain_available
    )
    
    self.whale_tracker = None
    self.network_health = None
    self.cascade_detector = None
    
    if is_onchain_available():
        try:
            onchain_config = {
                'santiment_api_key': os.getenv('SANTIMENT_API_KEY'),
                'dune_api_key': os.getenv('DUNE_API_KEY'),
                'etherscan_api_key': os.getenv('ETHERSCAN_API_KEY'),
                'update_interval': 60,
            }
            self.whale_tracker = OnChainWhaleTracker(onchain_config)
            self.network_health = OnChainNetworkHealth(onchain_config)
            self.cascade_detector = EnhancedCascadeDetector()
            
            self.metrics['onchain_enabled'] = True
            logger.info("On-chain primitives initialized for Redis publishing")
        except Exception as e:
            logger.warning(f"On-chain initialization failed: {e}")
            self.metrics['onchain_enabled'] = False
    else:
        self.metrics['onchain_enabled'] = False


def publish_onchain_signals(self, symbol: str = 'BTCUSDT'):
    """
    Fetch on-chain signals and publish to Redis.
    
    Called periodically (every 60 seconds) from the main run loop.
    
    Args:
        symbol: Trading symbol for Redis key namespace
    """
    if not self.whale_tracker:
        return
    
    import time
    current_ts = int(time.time())
    
    try:
        # === WHALE ACTIVITY SIGNALS ===
        whale_signals = self.whale_tracker.update(current_ts)
        
        # Publish whale signals to Redis
        whale_keys = {
            f'onchain:{symbol}:exchange_netflow': whale_signals.get('exchange_netflow', 0),
            f'onchain:{symbol}:netflow_zscore': whale_signals.get('exchange_netflow_zscore', 0),
            f'onchain:{symbol}:whale_pressure': whale_signals.get('whale_pressure', 0),
            f'onchain:{symbol}:large_tx_count': whale_signals.get('large_tx_count_5min', 0),
            f'onchain:{symbol}:concentration_risk': whale_signals.get('concentration_risk', 0),
        }
        
        # === NETWORK HEALTH SIGNALS ===
        if self.network_health:
            health_signals = self.network_health.update(current_ts)
            
            whale_keys.update({
                f'onchain:{symbol}:active_addresses_zscore': health_signals.get('active_addresses_zscore', 0),
                f'onchain:{symbol}:hash_rate_health': health_signals.get('hash_rate_health', 1.0),
                f'onchain:{symbol}:fee_pressure': health_signals.get('fee_pressure', 0),
                f'onchain:{symbol}:holder_conviction': health_signals.get('holder_conviction', 0.5),
                f'onchain:{symbol}:network_health_score': health_signals.get('network_health_score', 1.0),
            })
        
        # === CASCADE RISK ===
        if self.cascade_detector:
            cascade_warning = self.cascade_detector.get_cascade_warning(
                funding=0.0,  # TODO: Get from futures data
                oi_change=0.0,
                vol_ratio=1.0,
                onchain=whale_signals
            )
            
            whale_keys.update({
                f'onchain:{symbol}:cascade_risk': cascade_warning['risk_score'],
                f'onchain:{symbol}:cascade_action': cascade_warning['action'],
                f'onchain:{symbol}:cascade_level': cascade_warning['risk_level'],
            })
            
            # Log high cascade risk
            if cascade_warning['risk_score'] > 0.4:
                logger.warning(
                    f"⚠️  CASCADE RISK: {cascade_warning['risk_level']} "
                    f"({cascade_warning['risk_score']:.2f}) - {cascade_warning['action']}"
                )
        
        # Publish all keys to Redis with 5-minute TTL
        pipeline = self.redis.pipeline()
        for key, value in whale_keys.items():
            if isinstance(value, str):
                pipeline.setex(key, 300, value)
            else:
                pipeline.setex(key, 300, str(value))
        pipeline.execute()
        
        # Update metrics
        self.metrics['onchain_signals_published'] = self.metrics.get('onchain_signals_published', 0) + 1
        self.metrics['onchain_last_update'] = current_ts
        
        logger.debug(f"Published {len(whale_keys)} on-chain signals to Redis")
    
    except Exception as e:
        logger.error(f"Failed to publish on-chain signals: {e}")
        self.metrics['onchain_publish_errors'] = self.metrics.get('onchain_publish_errors', 0) + 1


# Add to the run() loop (called every 60 seconds):
# if int(time.time()) % 60 == 0:
#     self.publish_onchain_signals('BTCUSDT')
