"""
HIMARI Systemic Risk Monitor (SRM) - Main Service

Main entry point for the SRM sidecar service.
Polls external APIs, calculates risk signals, and publishes to Redis.

Usage:
    python -m srm.main

Environment Variables:
    SRM_REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    SRM_SYMBOLS: Comma-separated symbols to monitor (default: BTCUSDT,ETHUSDT)
    SRM_DRY_RUN: Enable dry-run mode (default: true)
    SRM_TELEGRAM_BOT_TOKEN: Telegram bot token for alerts
    SRM_TELEGRAM_CHAT_ID: Telegram chat ID for alerts
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Optional, Any
import logging

from .config import SRMConfig
from .signals import (
    FundingSaturationIndex,
    LiquidityEvaporationIndex,
    OracleDivergenceScore,
    StablecoinStressIndex,
    LeverageConcentrationIndex,
    CrossAssetContagionIndex,
)
from .composite import CompositeRiskCalculator, CompositeRiskResult
from .guardian import SystemicRiskGuardian, EmergencyExitExecutor, RiskAction, RiskDecision
from .services import SRMRedisClient, RateLimiter
from .alerts import AlertManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('himari.srm')


class SRMService:
    """
    Main SRM Service.
    
    Orchestrates all signal calculations, composite risk assessment,
    and guardian responses. Runs as an asynchronous sidecar service
    polling external APIs at configurable intervals.
    
    Attributes:
        config: SRM configuration
        redis: Redis client for publishing risk scores
        rate_limiter: Token bucket rate limiter for APIs
        signals: Per-symbol signal calculator instances
        calculators: Per-symbol composite risk calculators
        guardian: Risk response decision maker
        alerts: Notification manager
        metrics: Runtime metrics and error tracking
    """
    
    def __init__(self, config: Optional[SRMConfig] = None) -> None:
        """
        Initialize SRM service with configuration.
        
        Args:
            config: SRM configuration. Loads from environment if not provided.
        
        Raises:
            RuntimeError: If Redis connection fails and is required
        """
        self.config: SRMConfig = config or SRMConfig.from_env()
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        
        # Initialize components
        logger.info("Initializing SRM components...")
        
        # Redis client with connectivity check
        self.redis: SRMRedisClient = SRMRedisClient(self.config.api.redis_url)
        if not self.redis.connected:
            logger.warning("Redis not connected - risk scores will not be published")
        
        # Rate limiter with API-specific configurations from config
        self.rate_limiter: RateLimiter = RateLimiter()
        self._configure_rate_limits()
        
        # Signals (one instance per signal type, shared across symbols)
        self.signals: Dict[str, Dict[str, Any]] = {}
        for symbol in self.config.symbols:
            self.signals[symbol] = {
                'fsi': FundingSaturationIndex(self.config.fsi),
                'lei': LiquidityEvaporationIndex(self.config.lei),
                'ods': OracleDivergenceScore(self.config.ods),
                'scsi': StablecoinStressIndex(self.config.scsi),
                'lci': LeverageConcentrationIndex(self.config.lci),
                'caci': CrossAssetContagionIndex(self.config.caci),
            }
        
        # Composite calculator (one per symbol)
        self.calculators: Dict[str, CompositeRiskCalculator] = {
            symbol: CompositeRiskCalculator(self.config.regime_lookback_minutes)
            for symbol in self.config.symbols
        }
        
        # Emergency executor
        self.executor: EmergencyExitExecutor = EmergencyExitExecutor(
            dry_run=self.config.guardian.dry_run
        )
        
        # Guardian
        self.guardian: SystemicRiskGuardian = SystemicRiskGuardian(
            redis_client=self.redis,
            emergency_executor=self.executor,
            reduce_threshold=self.config.guardian.reduce_threshold,
            close_only_threshold=self.config.guardian.close_only_threshold,
            halt_threshold=self.config.guardian.halt_threshold,
            confirmation_count=self.config.guardian.confirmation_count,
        )
        
        # Alert manager
        self.alerts: AlertManager = AlertManager(
            telegram_bot_token=self.config.telegram_bot_token,
            telegram_chat_id=self.config.telegram_chat_id,
        )
        
        # Enhanced metrics with error type tracking including timestamps
        self.metrics: Dict[str, Any] = {
            'cycles': 0,
            'errors': 0,
            'error_types': {},  # Track error types: {key: {'count': N, 'first_seen': ts, 'last_seen': ts}}
            'last_update': None,
            'signals_calculated': 0,
            'emergency_exits': 0,
            'start_time': datetime.utcnow().isoformat(),
        }
        
        # Validate regime weights on startup
        self._validate_regime_weights()
        
        # Log startup banner
        self._log_startup_banner()
    
    def _log_startup_banner(self) -> None:
        """Log prominent startup banner with configuration summary."""
        banner = [
            "",
            "=" * 60,
            "  HIMARI SYSTEMIC RISK MONITOR (SRM)",
            "=" * 60,
        ]
        
        if self.config.guardian.dry_run:
            banner.extend([
                "",
                "  ⚠️  DRY-RUN MODE ENABLED ⚠️",
                "  Emergency exits will be LOGGED but NOT EXECUTED",
                "",
            ])
        else:
            banner.extend([
                "",
                "  🔴 LIVE MODE - EMERGENCY EXITS WILL EXECUTE 🔴",
                "",
            ])
        
        banner.extend([
            f"  Symbols: {', '.join(self.config.symbols)}",
            f"  Redis: {self.config.api.redis_url}",
            f"  Redis Connected: {self.redis.connected}",
            f"  Thresholds: Reduce={self.config.guardian.reduce_threshold}, "
            f"Close={self.config.guardian.close_only_threshold}, "
            f"Halt={self.config.guardian.halt_threshold}",
            "=" * 60,
            "",
        ])
        
        for line in banner:
            logger.info(line)
    
    def _validate_regime_weights(self) -> None:
        """Validate that all regime weights are properly configured."""
        from .regime import REGIME_WEIGHTS, MarketRegime
        
        for regime in MarketRegime:
            if regime not in REGIME_WEIGHTS:
                logger.warning(f"No weights configured for regime {regime.value}")
                continue
            
            weights = REGIME_WEIGHTS[regime]
            if not weights.validate():
                logger.warning(f"Regime weights validation failed for {regime.value}")
            else:
                logger.debug(f"Regime weights validated for {regime.value}")
    
    def _configure_rate_limits(self) -> None:
        """Configure rate limiter with API-specific limits from config."""
        # Use API config for endpoint-specific rate limits
        api_limits = {
            'binance': {'max_tokens': 1200, 'refill_rate': 20.0},  # 1200/min
            'coingecko': {'max_tokens': 10, 'refill_rate': 0.17},  # 10/min (free tier)
            'coinglass': {'max_tokens': 30, 'refill_rate': 0.5},   # 30/min
            'yahoo': {'max_tokens': 100, 'refill_rate': 2.0},      # Generous limit
            'coinbase': {'max_tokens': 10, 'refill_rate': 0.17},   # ~10/min
            'kraken': {'max_tokens': 15, 'refill_rate': 0.25},     # 15/min
        }
        
        for api_name, limits in api_limits.items():
            self.rate_limiter.configure(api_name, limits['max_tokens'], limits['refill_rate'])
        
        logger.debug("Rate limiter configured for all APIs")
    
    def _track_error(self, error_type: str, error: Exception) -> None:
        """
        Track error occurrence for debugging and monitoring.
        
        Args:
            error_type: Category of error (e.g., 'fsi', 'redis', 'api')
            error: The exception that occurred
        """
        self.metrics['errors'] += 1
        error_key = f"{error_type}:{type(error).__name__}"
        now = datetime.utcnow().isoformat()
        
        if error_key not in self.metrics['error_types']:
            self.metrics['error_types'][error_key] = {
                'count': 1,
                'first_seen': now,
                'last_seen': now,
            }
        else:
            self.metrics['error_types'][error_key]['count'] += 1
            self.metrics['error_types'][error_key]['last_seen'] = now
    
    async def _calculate_signals(self, symbol: str) -> Dict[str, float]:
        """
        Calculate all enabled signals for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        
        Returns:
            Dict mapping signal names to scores (0.0-1.0)
        """
        signals = self.signals[symbol]
        results: Dict[str, float] = {}
        
        # Determine volatility regime for FSI using CACI VIX data as proxy
        tradfi_data: Dict[str, Optional[float]] = await signals['caci'].fetch_tradfi_data()
        vix: Optional[float] = tradfi_data.get('vix')
        vol_regime: str
        if vix is not None:
            if vix < 20:
                vol_regime = 'LOW'
            elif vix > 30:
                vol_regime = 'HIGH'
            else:
                vol_regime = 'MEDIUM'
        else:
            vol_regime = 'MEDIUM'
        
        # FSI: Funding Saturation Index
        if self.config.enable_fsi:
            try:
                await self.rate_limiter.wait_and_acquire('binance')
                await signals['fsi'].refresh_from_api(symbol)
                score, meta = signals['fsi'].calculate(vol_regime)
                results['fsi'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"FSI({symbol}): {score:.3f} - {meta.get('direction', 'N/A')}")
            except Exception as e:
                logger.error(f"FSI calculation failed: {e}")
                self._track_error('fsi', e)
                results['fsi'] = 0.0
        
        # LEI: Liquidity Evaporation Index
        if self.config.enable_lei:
            try:
                await self.rate_limiter.wait_and_acquire('binance', 5)  # Weight=5 for order book
                score, meta = await signals['lei'].refresh_from_api(symbol)
                results['lei'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"LEI({symbol}): {score:.3f} - evap={meta.get('evaporation_pct', 0):.1f}%")
            except Exception as e:
                logger.error(f"LEI calculation failed: {e}")
                self._track_error('lei', e)
                results['lei'] = 0.0
        
        # ODS: Oracle Divergence Score
        if self.config.enable_ods:
            try:
                await self.rate_limiter.wait_and_acquire('binance')
                await self.rate_limiter.wait_and_acquire('coingecko')
                base_symbol: str = symbol.replace('USDT', '').replace('USD', '')
                score, meta = await signals['ods'].refresh_from_api(base_symbol)
                results['ods'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"ODS({symbol}): {score:.3f} - divergence={meta.get('divergence_pct', 0):.2f}%")
            except Exception as e:
                logger.error(f"ODS calculation failed: {e}")
                self._track_error('ods', e)
                results['ods'] = 0.0
        
        # SCSI: Stablecoin Stress Index
        if self.config.enable_scsi:
            try:
                await self.rate_limiter.wait_and_acquire('coingecko')
                score, meta = await signals['scsi'].refresh_from_api()
                results['scsi'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"SCSI: {score:.3f} - type={meta.get('failure_type', 'none')}")
            except Exception as e:
                logger.error(f"SCSI calculation failed: {e}")
                self._track_error('scsi', e)
                results['scsi'] = 0.0
        
        # LCI: Leverage Concentration Index
        if self.config.enable_lci:
            try:
                await self.rate_limiter.wait_and_acquire('binance')
                base_symbol = symbol.replace('USDT', '').replace('USD', '')
                score, meta = await signals['lci'].refresh_from_api(base_symbol)
                results['lci'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"LCI({symbol}): {score:.3f} - HHI={meta.get('hhi', 0):.0f}")
            except Exception as e:
                logger.error(f"LCI calculation failed: {e}")
                self._track_error('lci', e)
                results['lci'] = 0.0
        
        # CACI: Cross-Asset Contagion Index
        if self.config.enable_caci:
            try:
                await self.rate_limiter.wait_and_acquire('yahoo')
                score, meta = signals['caci'].calculate(tradfi_data)
                results['caci'] = score
                self.metrics['signals_calculated'] += 1
                logger.debug(f"CACI: {score:.3f} - level={meta.get('stress_level', 'UNKNOWN')}")
            except Exception as e:
                logger.error(f"CACI calculation failed: {e}")
                self._track_error('caci', e)
                results['caci'] = 0.0
        
        return results
    
    async def _process_symbol(self, symbol: str) -> None:
        """
        Process a single symbol through the complete SRM pipeline.
        
        Pipeline stages:
        1. Calculate all enabled signals
        2. Compute composite risk with regime-aware weighting
        3. Publish to Redis
        4. Evaluate guardian response
        5. Execute emergency actions if required
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        """
        try:
            # Calculate all signals
            signal_values: Dict[str, float] = await self._calculate_signals(symbol)
            
            # Calculate composite risk
            calculator: CompositeRiskCalculator = self.calculators[symbol]
            result: CompositeRiskResult = calculator.calculate(signal_values, datetime.utcnow())
            
            # Publish to Redis
            if self.redis.connected:
                self.redis.publish_risk(symbol, result)
            
            # Evaluate guardian response
            decision: RiskDecision = self.guardian.evaluate(symbol)
            
            # Log summary
            logger.info(
                f"SRM {symbol}: score={result.score:.3f} "
                f"regime={result.regime.value} "
                f"action={decision.action.value}"
            )
            
            # Handle emergency actions
            if decision.requires_emergency_exit:
                self.metrics['emergency_exits'] += 1
                await self.alerts.send_emergency_alert(
                    symbol, result.score, result.regime.value, "HALT"
                )
                exit_result: Dict[str, Any] = await self.guardian.execute_emergency_exit(symbol)
                logger.critical(f"Emergency exit result: {exit_result}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self._track_error('process_symbol', e)
    
    async def _verify_redis_connection(self) -> bool:
        """
        Verify Redis connectivity before entering main loop.
        
        If reconnection is needed, also updates the Guardian's reference
        to use the new Redis client.
        
        Returns:
            True if Redis is connected, False otherwise
        """
        if not self.redis.connected:
            logger.warning("Redis not connected, attempting reconnection...")
            try:
                self.redis = SRMRedisClient(self.config.api.redis_url)
                if self.redis.connected:
                    # Update Guardian's reference to new Redis client
                    self.guardian.redis_client = self.redis
                    logger.info("Redis reconnected successfully")
                return self.redis.connected
            except Exception as e:
                logger.error(f"Redis reconnection failed: {e}")
                return False
        return True
    
    async def run(self) -> None:
        """
        Main processing loop.
        
        Continuously polls signals and updates risk scores until
        shutdown is requested. Handles graceful shutdown on SIGINT/SIGTERM.
        """
        self._running = True
        
        # Setup signal handlers for graceful shutdown
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
        
        # Verify Redis connectivity before starting
        await self._verify_redis_connection()
        
        logger.info("SRM service starting...")
        
        try:
            while self._running and not self._shutdown_event.is_set():
                cycle_start: datetime = datetime.utcnow()
                
                # Process all symbols
                for symbol in self.config.symbols:
                    if self._shutdown_event.is_set():
                        break
                    await self._process_symbol(symbol)
                
                self.metrics['cycles'] += 1
                self.metrics['last_update'] = datetime.utcnow().isoformat()
                
                # Log periodic status (every minute at 5s interval)
                if self.metrics['cycles'] % 12 == 0:
                    logger.info(
                        f"SRM status: cycles={self.metrics['cycles']} "
                        f"signals={self.metrics['signals_calculated']} "
                        f"errors={self.metrics['errors']} "
                        f"emergency_exits={self.metrics['emergency_exits']}"
                    )
                    # Log error breakdown if errors occurred
                    if self.metrics['error_types']:
                        logger.debug(f"Error breakdown: {self.metrics['error_types']}")
                
                # Wait for next cycle
                elapsed: float = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time: float = max(0, self.config.main_loop_interval - elapsed)
                
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop
                
        except Exception as e:
            logger.error(f"SRM service error: {e}")
            raise
        finally:
            await self.shutdown()
    
    def _shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()
    
    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("SRM service shutting down...")
        self._running = False
        self.redis.close()
        logger.info("SRM service stopped")


async def main() -> None:
    """Entry point for SRM service."""
    service: SRMService = SRMService()
    await service.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(0)
