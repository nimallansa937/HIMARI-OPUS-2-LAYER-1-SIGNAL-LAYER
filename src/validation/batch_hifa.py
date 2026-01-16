"""
Batch Processing for HIFA Validation

Optimizes throughput by running early stages on all candidates in batch,
then selectively running expensive stages on survivors.
"""

import time
from typing import List, Dict, Optional
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.genome import StrategyGenome
from .hifa import HIFAPipeline, ValidationReport

logger = logging.getLogger(__name__)


class BatchHIFAProcessor:
    """
    Batch processing for HIFA validation.

    Optimization: Run stages 0-2 on all candidates in batch,
    then selectively run expensive stages 3-6 on top-K survivors.
    """

    def __init__(
        self,
        pipeline: HIFAPipeline,
        batch_size: int = 100,
        top_k_surrogate: int = 20,
        parallel_backtests: int = 4
    ):
        """
        Args:
            pipeline: HIFA pipeline instance
            batch_size: Max candidates to process at once
            top_k_surrogate: How many to pass from surrogate to backtest
            parallel_backtests: Concurrent backtest threads
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.top_k = top_k_surrogate
        self.parallel_backtests = parallel_backtests

    def process_batch(
        self,
        candidates: List[StrategyGenome]
    ) -> List[ValidationReport]:
        """
        Process batch through HIFA with early filtering.

        Pipeline:
        1. Grammar validation (all, parallel)
        2. DSR gate (all, sequential for trial counting)
        3. Surrogate ranking (batch inference)
        4. Select top-K by predicted Sharpe
        5. Full validation (expensive stages) on survivors

        Returns:
            List of ValidationReport for all candidates
        """
        start_time = time.time()
        results = []

        # Split into batches if needed
        for batch_start in range(0, len(candidates), self.batch_size):
            batch = candidates[batch_start:batch_start + self.batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)

        logger.info(f"Batch HIFA processed {len(candidates)} candidates in "
                   f"{(time.time() - start_time):.2f}s")

        return results

    def _process_single_batch(
        self,
        candidates: List[StrategyGenome]
    ) -> List[ValidationReport]:
        """Process a single batch through HIFA."""
        results = []

        # Stage 0: Grammar validation (fast, can parallelize)
        grammar_results = self._batch_grammar(candidates)
        grammar_passed = [
            c for c, passed in zip(candidates, grammar_results) if passed
        ]

        # Add failed grammar results
        for c, passed in zip(candidates, grammar_results):
            if not passed:
                results.append(self._create_failed_report(c, "grammar", "Grammar validation failed"))

        if not grammar_passed:
            return results

        # Stage 1: DSR gate (needs sequential for trial counting)
        dsr_passed = []
        for candidate in grammar_passed:
            dsr_result = self.pipeline._stage1_dsr(candidate)
            if dsr_result.passed:
                dsr_passed.append(candidate)
            else:
                results.append(self._create_failed_report(
                    candidate, "dsr", dsr_result.reason
                ))

        if not dsr_passed:
            return results

        # Stage 2: Surrogate ranking (batch inference)
        surrogate_scores = self._batch_surrogate(dsr_passed)

        # Select top-K
        scored_candidates = list(zip(dsr_passed, surrogate_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = [c for c, _ in scored_candidates[:self.top_k]]

        # Add non-selected as failed at surrogate
        for c, score in scored_candidates[self.top_k:]:
            results.append(self._create_failed_report(
                c, "surrogate", f"Not in top-{self.top_k} (score={score:.2f})"
            ))

        # Stages 3-6: Full validation on top-K (parallel)
        full_results = self._parallel_full_validation(top_k)
        results.extend(full_results)

        return results

    def _batch_grammar(
        self,
        candidates: List[StrategyGenome]
    ) -> List[bool]:
        """Batch grammar validation."""
        results = []
        for c in candidates:
            is_valid, _ = self.pipeline.grammar.validate_genome(c)
            results.append(is_valid)
        return results

    def _batch_surrogate(
        self,
        candidates: List[StrategyGenome]
    ) -> List[float]:
        """
        Batch surrogate inference.

        Returns predicted Sharpe for each candidate.
        """
        if not candidates:
            return []

        try:
            # Stack vectors for batch inference
            vectors = torch.stack([
                torch.tensor(c.to_vector(), dtype=torch.float32)
                for c in candidates
            ])

            with torch.no_grad():
                predictions = self.pipeline.surrogate(vectors)

            # Extract Sharpe predictions
            scores = predictions[:, 0].numpy().tolist()
            return scores

        except Exception as e:
            logger.warning(f"Batch surrogate failed: {e}")
            # Fallback to random scores
            return [np.random.uniform(1.0, 2.5) for _ in candidates]

    def _parallel_full_validation(
        self,
        candidates: List[StrategyGenome]
    ) -> List[ValidationReport]:
        """
        Run full validation (stages 3-6) in parallel.

        Uses thread pool for parallel backtesting.
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_backtests) as executor:
            future_to_candidate = {
                executor.submit(self._run_remaining_stages, c): c
                for c in candidates
            }

            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    report = future.result()
                    results.append(report)
                except Exception as e:
                    logger.error(f"Validation failed for {candidate.id}: {e}")
                    results.append(self._create_failed_report(
                        candidate, "fast_backtest", str(e)
                    ))

        return results

    def _run_remaining_stages(
        self,
        candidate: StrategyGenome
    ) -> ValidationReport:
        """Run stages 3-6 for a single candidate."""
        start_time = time.time()
        stages_passed = ["grammar", "dsr", "surrogate"]
        all_results = {}

        # Stage 3: Fast backtest
        result = self.pipeline._stage3_fast_backtest(candidate)
        all_results['fast_backtest'] = result
        if not result.passed:
            return self._build_report(
                candidate, stages_passed, "fast_backtest",
                result, all_results, start_time
            )
        stages_passed.append("fast_backtest")

        # Stage 4: Full backtest
        result = self.pipeline._stage4_full_backtest(candidate)
        all_results['full_backtest'] = result
        if not result.passed:
            return self._build_report(
                candidate, stages_passed, "full_backtest",
                result, all_results, start_time
            )
        stages_passed.append("full_backtest")

        # Stage 5: True contribution
        result = self.pipeline._stage5_true_contribution(candidate)
        all_results['true_contribution'] = result
        if not result.passed:
            return self._build_report(
                candidate, stages_passed, "true_contribution",
                result, all_results, start_time
            )
        stages_passed.append("true_contribution")

        # Stage 6: Neutralization
        result = self.pipeline._stage6_neutralization(candidate)
        all_results['neutralization'] = result
        if not result.passed:
            return self._build_report(
                candidate, stages_passed, "neutralization",
                result, all_results, start_time
            )
        stages_passed.append("neutralization")

        # All passed!
        return self._build_report(
            candidate, stages_passed, "approved",
            result, all_results, start_time
        )

    def _create_failed_report(
        self,
        candidate: StrategyGenome,
        stage: str,
        reason: str
    ) -> ValidationReport:
        """Create a failed validation report."""
        from .hifa import HIFAResult

        stages_before = {
            'grammar': [],
            'dsr': ['grammar'],
            'surrogate': ['grammar', 'dsr'],
            'fast_backtest': ['grammar', 'dsr', 'surrogate'],
            'full_backtest': ['grammar', 'dsr', 'surrogate', 'fast_backtest'],
            'true_contribution': ['grammar', 'dsr', 'surrogate', 'fast_backtest', 'full_backtest'],
            'neutralization': ['grammar', 'dsr', 'surrogate', 'fast_backtest', 'full_backtest', 'true_contribution']
        }

        return ValidationReport(
            strategy_id=candidate.id,
            stages_passed=stages_before.get(stage, []),
            final_stage=stage,
            final_result=HIFAResult(
                passed=False,
                score=0.0,
                metrics={},
                reason=reason,
                latency_ms=0,
                stage_name=stage
            ),
            total_latency_ms=0,
            approved=False,
            approval_confidence=0,
            all_results={}
        )

    def _build_report(
        self,
        candidate: StrategyGenome,
        stages_passed: List[str],
        final_stage: str,
        final_result,
        all_results: Dict,
        start_time: float
    ) -> ValidationReport:
        """Build complete validation report."""
        return ValidationReport(
            strategy_id=candidate.id,
            stages_passed=stages_passed,
            final_stage=final_stage,
            final_result=final_result,
            total_latency_ms=(time.time() - start_time) * 1000,
            approved=final_stage == "approved",
            approval_confidence=final_result.score / 3.0 if final_stage == "approved" else 0,
            all_results=all_results
        )


class StreamingHIFAProcessor:
    """
    Streaming processor for continuous candidate validation.

    Maintains a queue of candidates and processes them as capacity allows.
    """

    def __init__(
        self,
        pipeline: HIFAPipeline,
        max_queue_size: int = 1000,
        process_interval_ms: int = 100
    ):
        self.pipeline = pipeline
        self.batch_processor = BatchHIFAProcessor(pipeline)
        self.max_queue_size = max_queue_size
        self.process_interval = process_interval_ms / 1000

        self.candidate_queue: List[StrategyGenome] = []
        self.result_queue: List[ValidationReport] = []
        self._running = False

    def add_candidates(self, candidates: List[StrategyGenome]) -> int:
        """
        Add candidates to processing queue.

        Returns:
            Number of candidates actually added
        """
        space_available = self.max_queue_size - len(self.candidate_queue)
        to_add = candidates[:space_available]
        self.candidate_queue.extend(to_add)
        return len(to_add)

    def process_pending(self, max_process: int = 100) -> List[ValidationReport]:
        """
        Process pending candidates and return results.

        Args:
            max_process: Maximum candidates to process in this call

        Returns:
            List of validation reports
        """
        if not self.candidate_queue:
            return []

        # Take batch from queue
        batch = self.candidate_queue[:max_process]
        self.candidate_queue = self.candidate_queue[max_process:]

        # Process
        results = self.batch_processor.process_batch(batch)
        self.result_queue.extend(results)

        return results

    def get_results(self, max_results: int = 100) -> List[ValidationReport]:
        """Get processed results."""
        results = self.result_queue[:max_results]
        self.result_queue = self.result_queue[max_results:]
        return results

    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            'pending_candidates': len(self.candidate_queue),
            'pending_results': len(self.result_queue)
        }
