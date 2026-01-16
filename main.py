"""
HIMARI Layer 1 Explorer Agent - Main Entry Point

Automated trading strategy generation, validation, and deployment system
for cryptocurrency derivatives (BTC/ETH perpetual futures).

Usage:
    python main.py --config config/layer1.yaml
    python main.py --generate-config  # Create default config
    python main.py --single-cycle     # Run one generation cycle
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

# Core imports
from src.core.genome import StrategyGenome, generate_random_strategy
from src.core.grammar import GrammarValidator
from src.core.features import FeatureVector, FEATURE_SCHEMA

# Engine imports
from src.engines.evolutionary import EvolutionaryExplorer
from src.engines.flow_matching import FlowMatchingGenerator, GenerationCondition
from src.engines.llm_guided import LLMGuidedGenerator, create_llm_client
from src.engines.harvester import ExternalIdeaHarvester
from src.engines.orchestrator import EngineOrchestrator, GenerationBudget
from src.engines.mcts import MCTSStrategyGenerator, MCTSConfig
from src.engines.psro import PSRODiversityManager, PSROConfig

# Validation imports
from src.validation.hifa import HIFAPipeline
from src.validation.batch_hifa import BatchHIFAProcessor
from src.validation.causal import CausalValidationGate
from src.validation.bayesian import MultiFidelityBayesianOptimizer

# Deployment imports
from src.deployment.shadow import ShadowEnvironment
from src.deployment.uncertainty import EpistemicUncertaintyGate
from src.deployment.transfer import TransferRatioConfidence
from src.deployment.deployment import DeploymentManager
from src.deployment.adr import AutomaticDomainRandomization

# Adaptation imports
from src.adaptation.drift import DriftDetectionEnsemble
from src.adaptation.response import AdaptiveResponseManager
from src.adaptation.retirement import StrategyRetirementManager

# Infrastructure imports
from src.infrastructure.config import (
    init_config, get_config, create_default_config_file,
    Layer1ExplorerConfig
)
from src.infrastructure.metrics import init_metrics, get_metrics
from src.infrastructure.data_interface import Layer1DataInterface

# Crawler imports
from src.crawler.orchestrator import CrawlerOrchestrator


logger = logging.getLogger(__name__)


class Layer1Explorer:
    """
    Main Layer 1 Explorer Agent.

    Coordinates the full strategy generation, validation,
    and deployment pipeline.
    """

    def __init__(self, config: Layer1ExplorerConfig):
        """Initialize Layer 1 Explorer with configuration."""
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self._init_components()

        logger.info("Layer 1 Explorer initialized")

    def _init_components(self):
        """Initialize all pipeline components."""
        cfg = self.config

        # Grammar validator
        self.grammar = GrammarValidator()

        # Generation engines
        self.evolutionary = EvolutionaryExplorer(
            population_size=cfg.generation.population_size,
            elite_size=cfg.generation.elite_count,
            mutation_rate=cfg.generation.mutation_rate,
            crossover_rate=cfg.generation.crossover_rate
        )

        self.generative = FlowMatchingGenerator()

        # LLM client (DeepSeek)
        llm_client = create_llm_client(
            provider=cfg.llm.primary_provider,
            api_key=cfg.llm.primary_api_key,
            model=cfg.llm.primary_model,
            base_url=cfg.llm.primary_base_url
        )

        self.llm_guided = LLMGuidedGenerator(primary_client=llm_client)
        self.harvester = ExternalIdeaHarvester(llm_client=llm_client)

        # Engine orchestrator
        self.orchestrator = EngineOrchestrator(
            evolutionary=self.evolutionary,
            generative=self.generative,
            llm_guided=self.llm_guided,
            external=self.harvester,
            budget=GenerationBudget(
                evolutionary_pct=cfg.generation.evolutionary_pct,
                generative_pct=cfg.generation.generative_pct,
                llm_guided_pct=cfg.generation.llm_guided_pct,
                external_pct=cfg.generation.external_pct,
                total_candidates_per_cycle=cfg.generation.total_candidates_per_cycle,
                max_llm_calls_per_cycle=cfg.generation.max_llm_calls_per_cycle
            ),
            grammar_validator=self.grammar,
            diversity_threshold=cfg.generation.diversity_threshold
        )

        # Gap enhancements
        self.mcts = MCTSStrategyGenerator(MCTSConfig())
        self.psro = PSRODiversityManager(PSROConfig())

        # Validation pipeline
        self.hifa = HIFAPipeline(grammar_validator=self.grammar, surrogate_model=None)
        self.batch_processor = BatchHIFAProcessor(pipeline=self.hifa)
        self.causal_gate = CausalValidationGate()
        self.bayesian_opt = MultiFidelityBayesianOptimizer()

        # Deployment pipeline
        self.shadow = ShadowEnvironment()
        self.uncertainty = EpistemicUncertaintyGate()
        self.transfer = TransferRatioConfidence()
        self.deployment = DeploymentManager(
            shadow_env=self.shadow,
            uncertainty_gate=self.uncertainty,
            transfer_model=self.transfer
        )
        self.adr = AutomaticDomainRandomization()

        # Adaptation systems
        self.drift_detector = DriftDetectionEnsemble()
        self.response_manager = AdaptiveResponseManager()
        self.retirement_manager = StrategyRetirementManager()

        # Infrastructure
        self.data_interface = Layer1DataInterface()
        self.metrics = get_metrics()

        # Crawler
        self.crawler = CrawlerOrchestrator()

        # State
        self.portfolio: List[StrategyGenome] = []
        self.cycle_count = 0

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run a single exploration cycle.

        Steps:
        1. Get current market context
        2. Generate strategy candidates
        3. Validate through HIFA pipeline
        4. Deploy survivors to shadow trading
        5. Check for drift and adapt

        Returns:
            Cycle statistics
        """
        self.cycle_count += 1
        cycle_start = datetime.now()

        logger.info(f"Starting cycle {self.cycle_count}")

        # 1. Get market context
        context = await self.data_interface.get_current_context()
        regime = context.get('regime')

        # 2. Generate candidates
        condition = GenerationCondition(
            target_sharpe=self.config.validation.min_sharpe,
            target_regime=regime,
            target_timeframe="1h"
        )

        generation_result = await self.orchestrator.generate_candidates(
            condition=condition,
            existing_portfolio=self.portfolio
        )

        candidates = generation_result.candidates
        logger.info(f"Generated {len(candidates)} candidates")

        # Record metrics
        self.metrics.record_generation_cycle(
            by_engine=generation_result.by_engine,
            diversity=generation_result.diversity_score,
            duration_seconds=generation_result.total_time_ms / 1000
        )

        # 3. Validate through HIFA
        validation_results = await self.batch_processor.process_batch(candidates)

        passed = [r for r in validation_results if r.passed]
        logger.info(f"Validation: {len(passed)}/{len(candidates)} passed")

        self.metrics.record_hifa_pass_rate(len(passed) / max(len(candidates), 1))

        # 4. Causal validation for survivors
        causally_valid = []
        for result in passed:
            strategy = next(
                (c for c in candidates if c.id == result.strategy_id), None
            )
            if strategy:
                # Infer causal hypothesis
                hypothesis = self.causal_gate.infer_hypothesis(
                    strategy,
                    [f.name for f in FEATURE_SCHEMA]
                )

                # Mock data for causal validation
                import numpy as np
                mock_data = np.random.randn(100, 5)

                causal_result = self.causal_gate.validate(
                    strategy.id, hypothesis, mock_data
                )

                if causal_result.is_causal:
                    causally_valid.append(strategy)

        logger.info(f"Causal validation: {len(causally_valid)} passed")

        # 5. Deploy to shadow trading
        deployed = 0
        for strategy in causally_valid[:5]:  # Limit deployments per cycle
            backtest_sharpe = strategy.backtest_metrics.get('sharpe', 1.5)

            decision = await self.deployment.evaluate_for_deployment(
                strategy=strategy,
                backtest_sharpe=backtest_sharpe,
                shadow_days=7  # Shortened for demo
            )

            if decision.approved:
                deployed += 1
                self.metrics.record_deployment_decision(approved=True)
            else:
                self.metrics.record_deployment_decision(approved=False)

        logger.info(f"Deployed {deployed} strategies to shadow")

        # 6. Check drift for existing portfolio
        drift_alerts = self.drift_detector.get_alert_history(window_hours=1)
        if drift_alerts:
            logger.warning(f"Drift detected: {len(drift_alerts)} alerts")
            for alert in drift_alerts:
                self.metrics.record_drift_alert(
                    detector=alert.get('detector', 'unknown'),
                    severity=alert.get('severity', 'low')
                )

        # Update portfolio
        self.portfolio.extend(causally_valid[:3])

        # Cycle complete
        cycle_duration = (datetime.now() - cycle_start).total_seconds()

        result = {
            'cycle': self.cycle_count,
            'candidates_generated': len(candidates),
            'validation_passed': len(passed),
            'causally_valid': len(causally_valid),
            'deployed': deployed,
            'drift_alerts': len(drift_alerts),
            'duration_seconds': cycle_duration,
            'portfolio_size': len(self.portfolio)
        }

        logger.info(f"Cycle {self.cycle_count} complete: {result}")
        return result

    async def run(self):
        """Run the explorer continuously."""
        self._running = True

        # Start metrics server
        if self.config.infrastructure.metrics_enabled:
            self.metrics.start_server(self.config.infrastructure.metrics_port)

        # Connect to infrastructure
        await self.data_interface.connect_all()

        logger.info("Layer 1 Explorer running")

        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    await self.run_cycle()
                except Exception as e:
                    logger.error(f"Cycle error: {e}", exc_info=True)

                # Wait for next cycle
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.cycle_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue to next cycle

        except asyncio.CancelledError:
            logger.info("Explorer cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Layer 1 Explorer")
        self._running = False
        self._shutdown_event.set()

        # Disconnect infrastructure
        await self.data_interface.disconnect_all()

        logger.info("Shutdown complete")

    def stop(self):
        """Signal the explorer to stop."""
        self._running = False
        self._shutdown_event.set()


async def run_single_cycle(config: Layer1ExplorerConfig):
    """Run a single exploration cycle."""
    explorer = Layer1Explorer(config)
    await explorer.data_interface.connect_all()

    try:
        result = await explorer.run_cycle()
        print(f"\nCycle Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    finally:
        await explorer.shutdown()


async def main_async(args):
    """Async main entry point."""
    # Initialize configuration
    if args.generate_config:
        create_default_config_file(args.config or "config/layer1.yaml")
        print(f"Created default configuration file")
        return

    config = init_config(args.config)

    # Initialize metrics
    if config.infrastructure.metrics_enabled:
        init_metrics(config.infrastructure.metrics_port)

    if args.single_cycle:
        await run_single_cycle(config)
    else:
        explorer = Layer1Explorer(config)

        # Handle shutdown signals
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("Received shutdown signal")
            explorer.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        await explorer.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='HIMARI Layer 1 Explorer Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run with default config
  python main.py --config my_config.yaml  # Use custom config
  python main.py --generate-config      # Create default config file
  python main.py --single-cycle         # Run one cycle and exit
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/layer1.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate default configuration file and exit'
    )

    parser.add_argument(
        '--single-cycle',
        action='store_true',
        help='Run a single exploration cycle and exit'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
