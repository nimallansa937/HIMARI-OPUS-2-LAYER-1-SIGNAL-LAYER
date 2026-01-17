"""
Hinance API Client for Layer 1 Explorer
Replaces MockShadowEnvironment with real Hinance Paper Trading integration.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import httpx

from ..core.genome import StrategyGenome
from .shadow import ShadowPerformance

logger = logging.getLogger(__name__)


class HinanceClient:
    """
    HTTP client for Layer 1 Explorer to communicate with Hinance Paper Trading.

    This class REPLACES the MockShadowEnvironment with real shadow trading.

    Usage in main.py:
        # OLD:
        self.shadow = ShadowEnvironment()  # Mock!

        # NEW:
        self.shadow = HinanceClient(
            hinance_url="http://localhost:8000",
            api_key=config.hinance_api_key
        )
    """

    def __init__(
        self,
        hinance_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Hinance API client.

        Args:
            hinance_url: Hinance API base URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.hinance_url = hinance_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

        # HTTP client
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        self.client = httpx.AsyncClient(
            base_url=self.hinance_url,
            headers=headers,
            timeout=timeout
        )

        logger.info(f"Hinance client initialized: {hinance_url}")

    async def run_shadow(
        self,
        strategy: StrategyGenome,
        duration_days: int = 21,
        capital: float = 10000
    ) -> ShadowPerformance:
        """
        Run strategy in Hinance shadow environment.

        This method has the SAME signature as MockShadowEnvironment.run_shadow()
        so it's a drop-in replacement.

        Args:
            strategy: Strategy genome to deploy
            duration_days: How many days to run shadow
            capital: Allocated capital

        Returns:
            ShadowPerformance metrics from real shadow trading
        """
        logger.info(
            f"Deploying strategy {strategy.id} to Hinance shadow. "
            f"Duration: {duration_days} days, Capital: ${capital:.2f}"
        )

        # 1. Deploy strategy to Hinance
        deploy_result = await self.deploy_strategy(
            strategy=strategy,
            capital=capital
        )

        if not deploy_result.get('success'):
            raise Exception(
                f"Failed to deploy strategy: {deploy_result.get('message')}"
            )

        logger.info(
            f"✅ Strategy {strategy.id} deployed to Hinance. "
            f"Allocated: ${deploy_result['allocated_capital']:.2f}"
        )

        # 2. Monitor shadow trading for duration
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)

        logger.info(
            f"Monitoring shadow trading until {end_date.strftime('%Y-%m-%d')}"
        )

        # Poll periodically (check every 6 hours)
        check_interval_hours = 6
        total_checks = (duration_days * 24) // check_interval_hours

        for check in range(total_checks):
            # Wait for check interval
            await asyncio.sleep(check_interval_hours * 3600)

            # Get current performance
            try:
                perf = await self.get_strategy_performance(strategy.id)

                logger.info(
                    f"Shadow day {(check + 1) * check_interval_hours / 24:.1f}: "
                    f"Sharpe={perf.get('sharpe_ratio', 0):.2f}, "
                    f"TR={perf.get('transfer_ratio', 0):.2f}, "
                    f"PnL=${perf.get('total_pnl', 0):.2f}"
                )

            except Exception as e:
                logger.error(f"Error checking performance: {e}")

        # 3. Get final performance
        logger.info(f"Shadow period complete. Fetching final performance...")

        final_perf = await self.get_strategy_performance(strategy.id)

        # 4. Convert to ShadowPerformance object
        shadow_performance = self._parse_performance(final_perf, strategy.id)

        logger.info(
            f"✅ Shadow trading complete for {strategy.id}. "
            f"Final Sharpe: {shadow_performance.sharpe_ratio:.2f}, "
            f"Transfer Ratio: {shadow_performance.transfer_ratio:.2f}"
        )

        return shadow_performance

    async def deploy_strategy(
        self,
        strategy: StrategyGenome,
        capital: float
    ) -> Dict[str, Any]:
        """
        Deploy strategy to Hinance shadow environment.

        Args:
            strategy: Strategy genome
            capital: Allocated capital

        Returns:
            Deployment response dict
        """
        # Serialize strategy genome
        genome_json = self._serialize_strategy(strategy)

        # Prepare request
        request_data = {
            'strategy_id': strategy.id,
            'genome_json': genome_json,
            'backtest_sharpe': strategy.backtest_metrics.get('sharpe', 1.5),
            'capital_allocation': capital,
            'metadata': {
                'deployed_by': 'layer1_explorer',
                'deployment_time': datetime.now().isoformat()
            }
        }

        try:
            response = await self.client.post(
                '/shadow/strategies/deploy',
                json=request_data
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Deployment failed: {e}")
            raise

    async def get_strategy_performance(
        self,
        strategy_id: str
    ) -> Dict[str, Any]:
        """
        Get strategy performance from Hinance.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance metrics dict
        """
        try:
            response = await self.client.get(
                f'/shadow/strategies/{strategy_id}/performance'
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Performance query failed: {e}")
            raise

    async def remove_strategy(
        self,
        strategy_id: str
    ) -> Dict[str, Any]:
        """
        Remove strategy from Hinance shadow environment.

        Args:
            strategy_id: Strategy to remove

        Returns:
            Removal response dict
        """
        try:
            response = await self.client.delete(
                f'/shadow/strategies/{strategy_id}'
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Strategy removal failed: {e}")
            raise

    async def list_strategies(self) -> Dict[str, Any]:
        """
        List all strategies in Hinance shadow.

        Returns:
            Strategy list dict
        """
        try:
            response = await self.client.get('/shadow/strategies')
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Strategy list query failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Hinance API is reachable.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get('/health')
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _serialize_strategy(self, strategy: StrategyGenome) -> str:
        """
        Serialize strategy genome to JSON.

        Args:
            strategy: Strategy genome

        Returns:
            JSON string
        """
        genome_dict = {
            'id': strategy.id,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'activation': node.activation.value,
                    'weights': node.weights,
                    'bias': node.bias,
                    'inputs': node.inputs
                }
                for node in strategy.nodes
            ],
            'input_connections': strategy.input_connections,
            'output_node': strategy.output_node,
            'base_position_pct': strategy.base_position_pct,
            'risk_threshold': strategy.risk_threshold,
            'backtest_metrics': strategy.backtest_metrics
        }

        return json.dumps(genome_dict)

    def _parse_performance(
        self,
        data: Dict[str, Any],
        strategy_id: str
    ) -> ShadowPerformance:
        """
        Parse Hinance performance response to ShadowPerformance object.

        Args:
            data: Performance dict from Hinance
            strategy_id: Strategy identifier

        Returns:
            ShadowPerformance object
        """
        return ShadowPerformance(
            strategy_id=strategy_id,
            start_date=datetime.fromisoformat(data['start_date']),
            end_date=datetime.fromisoformat(data['end_date']),
            total_pnl=data['total_pnl'],
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'],
            trade_count=data['trade_count'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            avg_trade_duration_hours=data['avg_trade_duration_hours'],
            transfer_ratio=data['transfer_ratio'],
            execution_residuals=data.get('execution_residuals', []),
            daily_returns=data.get('daily_returns', [])
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("Hinance client closed")


# ============================================================================
# Compatibility Layer
# ============================================================================

class HinanceShadowEnvironment(HinanceClient):
    """
    Alias for backward compatibility with ShadowEnvironment interface.

    This class has the exact same interface as the old MockShadowEnvironment,
    so it's a drop-in replacement in Layer 1's main.py.
    """
    pass
