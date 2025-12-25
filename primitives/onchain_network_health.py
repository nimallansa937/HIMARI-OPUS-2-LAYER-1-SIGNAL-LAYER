"""
OnChain Network Health Monitor
==============================

Network-level health indicators for regime detection and cascade prediction.

Monitors active addresses, hash rate, and network congestion to provide
early warning signals for network stress and holder behavior patterns.

Data Sources:
- Blockchain.com API (Bitcoin network stats)
- Etherscan API (Ethereum gas/network stats)

Update Cadence: 60 seconds
Latency Impact: <1ms (cached results)

Output Signals:
- active_addresses_zscore: [-3, +3] User adoption trend
- hash_rate_health: [0, 1] Mining network stability
- fee_pressure: [0, 1] Network congestion level
- holder_conviction: [0, 1] HODLer strength indicator
"""

import os
import time
import logging
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import requests

logger = logging.getLogger(__name__)


class OnChainNetworkHealth:
    """
    Network-level health indicators for regime detection.
    
    Tracks network activity, mining health, and fee pressure to identify
    stress conditions and regime transitions.
    
    Feeds into: RegimeLayer, CascadeDetector, Signal Fusion
    
    Example:
        health = OnChainNetworkHealth(config)
        signals = health.update(timestamp)
        
        if signals['fee_pressure'] > 0.8:
            logger.warning("Network congestion detected!")
    """
    
    def __init__(self, config: Dict):
        """
        Initialize network health monitor.
        
        Args:
            config: Configuration dict with API keys and settings
        """
        self.etherscan_key = config.get('etherscan_api_key') or os.getenv('ETHERSCAN_API_KEY', 'X3NUGJTD8MIW3K8UCSM8T4TT6CE71AYPIK')
        self.update_interval = config.get('update_interval', 60)  # seconds
        
        # Baseline statistics (for z-score computation)
        self.baseline_active_addresses = None
        self.baseline_hash_rate = None
        self.baseline_fee = None
        
        # Running statistics (Welford algorithm)
        self.addr_count = 0
        self.addr_mean = 0.0
        self.addr_m2 = 0.0
        
        # Cache
        self.last_update = 0
        self._cached_signals = {
            'active_addresses_zscore': 0.0,
            'hash_rate_health': 1.0,
            'fee_pressure': 0.0,
            'holder_conviction': 0.5,
            'network_health_score': 1.0,
        }
        
        logger.info("OnChainNetworkHealth initialized")
    
    def update(self, timestamp: int) -> Dict[str, float]:
        """
        Fetch latest network health data and compute signals.
        
        Returns cached results if called within update_interval.
        
        Args:
            timestamp: Current Unix timestamp (seconds)
            
        Returns:
            Dict of network health signals
        """
        if timestamp - self.last_update < self.update_interval:
            return self._cached_signals
        
        try:
            # Fetch Bitcoin network stats from Blockchain.com
            btc_stats = self._fetch_bitcoin_network_stats()
            
            # Fetch Ethereum network stats from Etherscan
            eth_stats = self._fetch_ethereum_network_stats()
            
            # Compute signals
            signals = {}
            
            # Active addresses (from Bitcoin)
            if btc_stats.get('n_transactions'):
                n_tx = btc_stats['n_transactions']
                
                # Update Welford statistics
                self._update_addr_stats(n_tx)
                
                # Compute z-score
                if self.addr_count > 10:
                    variance = self.addr_m2 / (self.addr_count - 1) if self.addr_count > 1 else 1
                    std = np.sqrt(variance) if variance > 0 else 1
                    zscore = (n_tx - self.addr_mean) / std
                    signals['active_addresses_zscore'] = float(np.clip(zscore / 3, -1, 1))
                else:
                    signals['active_addresses_zscore'] = 0.0
            else:
                signals['active_addresses_zscore'] = self._cached_signals['active_addresses_zscore']
            
            # Hash rate health (from Bitcoin)
            if btc_stats.get('hash_rate') and self.baseline_hash_rate:
                hash_ratio = btc_stats['hash_rate'] / self.baseline_hash_rate
                signals['hash_rate_health'] = float(np.clip(hash_ratio, 0, 1))
            elif btc_stats.get('hash_rate'):
                # Set baseline on first run
                self.baseline_hash_rate = btc_stats['hash_rate']
                signals['hash_rate_health'] = 1.0
            else:
                signals['hash_rate_health'] = self._cached_signals['hash_rate_health']
            
            # Fee pressure (from Ethereum - more dynamic than BTC)
            if eth_stats.get('gas_price'):
                gas_gwei = eth_stats['gas_price']
                # Normalize: 20 gwei = low, 100 gwei = high pressure
                fee_pressure = np.clip((gas_gwei - 20) / 80, 0, 1)
                signals['fee_pressure'] = float(fee_pressure)
            else:
                signals['fee_pressure'] = self._cached_signals['fee_pressure']
            
            # Holder conviction (simplified - based on network activity)
            # High activity = low conviction (selling), low activity = high conviction (HODLing)
            if btc_stats.get('n_transactions') and self.addr_mean > 0:
                activity_ratio = btc_stats['n_transactions'] / self.addr_mean
                conviction = 1.0 / (1.0 + activity_ratio)  # Inverse relationship
                signals['holder_conviction'] = float(np.clip(conviction, 0, 1))
            else:
                signals['holder_conviction'] = 0.5
            
            # Overall network health score
            health = (
                (1.0 - abs(signals['active_addresses_zscore'])) * 0.3 +
                signals['hash_rate_health'] * 0.4 +
                (1.0 - signals['fee_pressure']) * 0.3
            )
            signals['network_health_score'] = float(np.clip(health, 0, 1))
            
            self._cached_signals = signals
            
            logger.debug(
                f"Network health: addresses_z={signals['active_addresses_zscore']:+.2f}, "
                f"hash_health={signals['hash_rate_health']:.2f}, "
                f"fee={signals['fee_pressure']:.2f}"
            )
        
        except Exception as e:
            logger.error(f"Failed to update network health: {e}")
        
        self.last_update = timestamp
        return self._cached_signals
    
    def _fetch_bitcoin_network_stats(self) -> Dict:
        """
        Fetch Bitcoin network statistics from Blockchain.com.
        
        Returns:
            Dict with 'n_transactions', 'hash_rate', 'difficulty'
        """
        try:
            # Get latest block for transaction count
            url = "https://blockchain.info/latestblock"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return {}
            
            latest = response.json()
            
            # Get stats
            stats_url = "https://api.blockchain.info/stats"
            stats_response = requests.get(stats_url, timeout=10)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                return {
                    'n_transactions': stats.get('n_tx', 0),
                    'hash_rate': stats.get('hash_rate', 0),
                    'difficulty': stats.get('difficulty', 0),
                    'estimated_tx_volume_btc': stats.get('estimated_btc_sent', 0) / 1e8,
                    'market_cap': stats.get('market_price_usd', 0) * 21_000_000,
                }
            
            return {'n_transactions': latest.get('n_tx', 0)}
        
        except Exception as e:
            logger.error(f"Blockchain.com stats error: {e}")
            return {}
    
    def _fetch_ethereum_network_stats(self) -> Dict:
        """
        Fetch Ethereum network statistics from Etherscan.
        
        Returns:
            Dict with 'gas_price', 'pending_tx_count', etc.
        """
        if not self.etherscan_key:
            return {}
        
        try:
            # Get current gas price using Etherscan V2 API
            url = "https://api.etherscan.io/v2/api"
            params = {
                'chainid': '1',  # Ethereum mainnet
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': self.etherscan_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            result = data.get('result', {})
            
            # Handle case where result is a string (error message)
            if isinstance(result, str):
                logger.warning(f"Etherscan returned message: {result}")
                return {}
            
            if result and isinstance(result, dict):
                return {
                    'gas_price': float(result.get('ProposeGasPrice', 20)),  # Gwei
                    'safe_gas': float(result.get('SafeGasPrice', 15)),
                    'fast_gas': float(result.get('FastGasPrice', 30)),
                    'base_fee': float(result.get('suggestBaseFee', 15)),
                }
            
            return {}
        
        except Exception as e:
            logger.error(f"Etherscan gas oracle error: {e}")
            return {}
    
    def _update_addr_stats(self, value: float) -> None:
        """Update running statistics using Welford's algorithm."""
        self.addr_count += 1
        delta = value - self.addr_mean
        self.addr_mean += delta / self.addr_count
        delta2 = value - self.addr_mean
        self.addr_m2 += delta * delta2
    
    def get_statistics(self) -> Dict:
        """Get current statistics for monitoring."""
        variance = self.addr_m2 / (self.addr_count - 1) if self.addr_count > 1 else 0
        
        return {
            'sample_count': self.addr_count,
            'mean_activity': self.addr_mean,
            'std_activity': np.sqrt(variance) if variance > 0 else 0,
            'baseline_hash_rate': self.baseline_hash_rate,
            'last_update': self.last_update,
        }


def create_network_health(etherscan_key: Optional[str] = None) -> OnChainNetworkHealth:
    """Factory function to create OnChainNetworkHealth instance."""
    config = {
        'etherscan_api_key': etherscan_key,
        'update_interval': 60,
    }
    return OnChainNetworkHealth(config)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("ONCHAIN NETWORK HEALTH TEST")
    print("=" * 70)
    
    # Create network health monitor
    health = create_network_health()
    
    print("\n" + "=" * 70)
    print("TEST 1: Fetch Network Statistics")
    print("=" * 70)
    
    # Test Bitcoin stats
    print("\nFetching Bitcoin network stats...")
    btc_stats = health._fetch_bitcoin_network_stats()
    if btc_stats:
        print(f"  Transactions (24h): {btc_stats.get('n_transactions', 'N/A'):,}")
        print(f"  Hash Rate: {btc_stats.get('hash_rate', 'N/A'):,.0f}")
        print(f"  Difficulty: {btc_stats.get('difficulty', 'N/A'):,.0f}")
    else:
        print("  Failed to fetch Bitcoin stats")
    
    # Test Ethereum stats
    print("\nFetching Ethereum network stats...")
    eth_stats = health._fetch_ethereum_network_stats()
    if eth_stats:
        print(f"  Gas Price: {eth_stats.get('gas_price', 'N/A')} Gwei")
        print(f"  Safe Gas: {eth_stats.get('safe_gas', 'N/A')} Gwei")
        print(f"  Fast Gas: {eth_stats.get('fast_gas', 'N/A')} Gwei")
    else:
        print("  Failed to fetch Ethereum stats")
    
    print("\n" + "=" * 70)
    print("TEST 2: Compute Health Signals")
    print("=" * 70)
    
    current_time = int(time.time())
    signals = health.update(current_time)
    
    print(f"\n📊 Network Health Signals:")
    print(f"  Active Addresses Z-Score: {signals['active_addresses_zscore']:+.3f}")
    print(f"  Hash Rate Health: {signals['hash_rate_health']:.3f}")
    print(f"  Fee Pressure: {signals['fee_pressure']:.3f}")
    print(f"  Holder Conviction: {signals['holder_conviction']:.3f}")
    print(f"  Overall Health Score: {signals['network_health_score']:.3f}")
    
    # Interpret results
    print(f"\n🔍 Interpretation:")
    if signals['fee_pressure'] > 0.7:
        print("  ⚠️  High network congestion")
    elif signals['fee_pressure'] < 0.3:
        print("  ✓ Low network congestion")
    else:
        print("  ~ Moderate network activity")
    
    if signals['hash_rate_health'] < 0.8:
        print("  ⚠️  Mining network showing weakness")
    else:
        print("  ✓ Mining network healthy")
    
    if signals['network_health_score'] > 0.7:
        print("  ✓ Overall: Network healthy")
    elif signals['network_health_score'] < 0.4:
        print("  ⚠️  Overall: Network stressed")
    else:
        print("  ~ Overall: Moderate network health")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
