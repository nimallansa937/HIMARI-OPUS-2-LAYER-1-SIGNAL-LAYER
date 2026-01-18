"""
Etherscan API Connector

FREE tier: 5 calls/second, 100K calls/day
Provides on-chain data for Ethereum:
- ETH price
- Transaction history
- Whale wallet tracking
- Gas prices
- Token transfers

Usage:
    eth = EtherscanConnector(api_key="YOUR_KEY")
    balance = eth.get_eth_balance("0x...")
    txs = eth.get_transactions("0x...")
    gas = eth.get_gas_price()
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class EtherscanConnector:
    """
    Etherscan FREE API connector for on-chain data.
    
    Rate Limits (free tier):
    - 5 calls/second
    - 100,000 calls/day
    
    Get free API key: https://etherscan.io/register
    
    Features:
    - ETH balances and transactions
    - Token balances and transfers
    - Gas price oracle
    - Contract ABIs
    - Whale wallet monitoring
    """
    
    # V2 API (V1 deprecated Aug 2025)
    BASE_URL = "https://api.etherscan.io/v2/api"
    CHAIN_ID = 1  # Ethereum mainnet
    
    # Popular whale/exchange wallets to monitor
    WHALE_WALLETS = {
        'binance_hot': '0x28C6c06298d514Db089934071355E5743bf21d60',
        'binance_cold': '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
        'coinbase': '0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43',
        'kraken': '0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2',
        'ftx_cold': '0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2',
    }
    
    def __init__(self, api_key: str):
        """
        Initialize Etherscan connector.
        
        Args:
            api_key: Etherscan API key (free tier available)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self._last_request_time = 0
        self._request_count = 0
    
    def _request(self, params: Dict) -> Dict:
        """Rate-limited request to Etherscan API."""
        # 5 calls/second = 200ms between requests
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 0.2:
            time.sleep(0.2 - time_since_last)
        
        params['apikey'] = self.api_key
        params['chainid'] = self.CHAIN_ID  # V2 API requires chainid
        response = self.session.get(self.BASE_URL, params=params)
        self._last_request_time = time.time()
        self._request_count += 1
        
        if response.status_code == 429:
            logger.warning("Rate limited by Etherscan, waiting 5s")
            time.sleep(5)
            return self._request(params)
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == '0' and data.get('message') != 'No transactions found':
            raise Exception(f"Etherscan API error: {data.get('message')}")
        
        return data
    
    # =========================================================================
    # ACCOUNT APIs
    # =========================================================================
    
    def get_eth_balance(self, address: str) -> float:
        """
        Get ETH balance for an address.
        
        Args:
            address: Ethereum address
            
        Returns:
            Balance in ETH
        """
        data = self._request({
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest',
        })
        
        # Convert from Wei to ETH
        wei = int(data.get('result', 0))
        return wei / 1e18
    
    def get_multi_balance(self, addresses: List[str]) -> Dict[str, float]:
        """
        Get ETH balance for multiple addresses (up to 20).
        
        Args:
            addresses: List of Ethereum addresses
            
        Returns:
            Dict mapping address -> balance in ETH
        """
        if len(addresses) > 20:
            raise ValueError("Maximum 20 addresses per request")
        
        data = self._request({
            'module': 'account',
            'action': 'balancemulti',
            'address': ','.join(addresses),
            'tag': 'latest',
        })
        
        return {
            item['account']: int(item['balance']) / 1e18
            for item in data.get('result', [])
        }
    
    def get_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get normal transactions for an address.
        
        Args:
            address: Ethereum address
            start_block: Start block number
            end_block: End block number
            page: Page number
            offset: Transactions per page (max 10000)
            
        Returns:
            List of transaction data
        """
        data = self._request({
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc',
        })
        
        txs = data.get('result', [])
        
        return [
            {
                'hash': tx['hash'],
                'block': int(tx['blockNumber']),
                'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),
                'from': tx['from'],
                'to': tx['to'],
                'value_eth': int(tx['value']) / 1e18,
                'gas_price_gwei': int(tx['gasPrice']) / 1e9,
                'gas_used': int(tx['gasUsed']),
                'is_error': tx['isError'] == '1',
            }
            for tx in txs if isinstance(tx, dict)
        ]
    
    def get_token_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get ERC-20 token transfers for an address.
        
        Args:
            address: Ethereum address
            contract_address: Filter by token contract (optional)
            page: Page number
            offset: Transfers per page
            
        Returns:
            List of token transfer data
        """
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'page': page,
            'offset': offset,
            'sort': 'desc',
        }
        
        if contract_address:
            params['contractaddress'] = contract_address
        
        data = self._request(params)
        
        return [
            {
                'hash': tx['hash'],
                'block': int(tx['blockNumber']),
                'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),
                'from': tx['from'],
                'to': tx['to'],
                'token_name': tx['tokenName'],
                'token_symbol': tx['tokenSymbol'],
                'token_decimals': int(tx['tokenDecimal']),
                'value': int(tx['value']) / (10 ** int(tx['tokenDecimal'])),
                'contract': tx['contractAddress'],
            }
            for tx in data.get('result', []) if isinstance(tx, dict)
        ]
    
    # =========================================================================
    # GAS & PRICE APIs
    # =========================================================================
    
    def get_gas_price(self) -> Dict[str, float]:
        """
        Get current gas prices.
        
        Returns:
            Dict with safe/propose/fast gas prices in Gwei
        """
        data = self._request({
            'module': 'gastracker',
            'action': 'gasoracle',
        })
        
        result = data.get('result', {})
        
        return {
            'safe_gwei': float(result.get('SafeGasPrice', 0)),
            'propose_gwei': float(result.get('ProposeGasPrice', 0)),
            'fast_gwei': float(result.get('FastGasPrice', 0)),
            'base_fee': float(result.get('suggestBaseFee', 0)),
        }
    
    def get_eth_price(self) -> Dict[str, float]:
        """
        Get current ETH price.
        
        Returns:
            Dict with ETH price in USD and BTC
        """
        data = self._request({
            'module': 'stats',
            'action': 'ethprice',
        })
        
        result = data.get('result', {})
        
        return {
            'eth_usd': float(result.get('ethusd', 0)),
            'eth_btc': float(result.get('ethbtc', 0)),
            'timestamp': datetime.fromtimestamp(int(result.get('ethusd_timestamp', 0))),
        }
    
    # =========================================================================
    # WHALE MONITORING
    # =========================================================================
    
    def get_whale_balances(self) -> Dict[str, Dict]:
        """
        Get ETH balances for known whale/exchange wallets.
        
        Returns:
            Dict mapping wallet_name -> {address, balance_eth}
        """
        addresses = list(self.WHALE_WALLETS.values())
        balances = self.get_multi_balance(addresses[:20])  # Max 20
        
        result = {}
        for name, addr in self.WHALE_WALLETS.items():
            result[name] = {
                'address': addr,
                'balance_eth': balances.get(addr, 0),
            }
        
        return result
    
    def watch_large_transfers(
        self,
        address: str,
        min_eth: float = 100.0,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get large ETH transfers for an address.
        
        Args:
            address: Wallet to monitor
            min_eth: Minimum transfer size in ETH
            limit: Max number of transfers to return
            
        Returns:
            List of large transfers
        """
        txs = self.get_transactions(address, offset=100)
        
        large_txs = [
            tx for tx in txs
            if tx['value_eth'] >= min_eth
        ]
        
        return large_txs[:limit]
    
    # =========================================================================
    # STATS APIs
    # =========================================================================
    
    def get_eth_supply(self) -> float:
        """Get total ETH supply."""
        data = self._request({
            'module': 'stats',
            'action': 'ethsupply',
        })
        
        return int(data.get('result', 0)) / 1e18
    
    def get_eth2_supply(self) -> Dict[str, float]:
        """Get ETH2 staking stats."""
        data = self._request({
            'module': 'stats',
            'action': 'ethsupply2',
        })
        
        result = data.get('result', {})
        
        return {
            'eth_supply': int(result.get('EthSupply', 0)) / 1e18,
            'eth2_staking': int(result.get('Eth2Staking', 0)) / 1e18,
            'burnt_fees': int(result.get('BurntFees', 0)) / 1e18,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Ethereum stats."""
        price = self.get_eth_price()
        gas = self.get_gas_price()
        
        return {
            'price_usd': price['eth_usd'],
            'price_btc': price['eth_btc'],
            'gas_safe_gwei': gas['safe_gwei'],
            'gas_fast_gwei': gas['fast_gwei'],
            'timestamp': datetime.now().isoformat(),
        }


# Quick test
if __name__ == "__main__":
    import os
    
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    
    if not api_key:
        print("Etherscan connector created.")
        print("Get free API key at: https://etherscan.io/register")
        print()
        print("Set API key:")
        print("  $env:ETHERSCAN_API_KEY='your-key-here'")
        print()
        print("Then test with:")
        print("  python -c \"from etherscan_connector import EtherscanConnector; ...")
    else:
        eth = EtherscanConnector(api_key=api_key)
        
        print("Testing Etherscan Connector...")
        
        # Get ETH price
        price = eth.get_eth_price()
        print(f"ETH Price: ${price['eth_usd']:,.2f}")
        
        # Get gas
        gas = eth.get_gas_price()
        print(f"Gas: {gas['safe_gwei']:.0f} / {gas['propose_gwei']:.0f} / {gas['fast_gwei']:.0f} Gwei")
        
        print("✓ Etherscan connector working!")
