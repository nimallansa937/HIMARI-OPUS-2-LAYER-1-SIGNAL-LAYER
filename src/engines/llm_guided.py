"""
Engine 3: LLM-Guided Strategy Generation

Performs semantic mutations—understanding the purpose of strategy components
rather than randomly swapping subtrees.

CRITICAL: LLMs operate OFFLINE, generating artifacts that execute deterministically.
No LLM calls in the real-time trading loop.

Supports multiple LLM backends:
- Gemini (primary)
- Ollama/DeepSeek (local)
- DeepSeek API (fallback)
"""

import asyncio
import json
import ast
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import httpx
import logging

from ..core.genome import StrategyGenome, SIGNAL_FEATURE_MAP
from .flow_matching import GenerationCondition

logger = logging.getLogger(__name__)


@dataclass
class LLMGeneratedStrategy:
    """A strategy generated or modified by LLM."""
    code: str
    explanation: str
    causal_hypothesis: str
    confidence: float
    raw_response: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-coder:33b",
        timeout: float = 60.0
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate using Ollama API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")


class GeminiClient(LLMClient):
    """Client for Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro"
    ):
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate using Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return ""


class DeepSeekClient(LLMClient):
    """Client for DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate using DeepSeek API."""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            return ""


class LLMGuidedGenerator:
    """
    LLM-augmented strategy generation via Chain-of-Alpha reasoning.

    Capabilities:
    1. Semantic mutation: Understand WHY strategy works, improve it
    2. Novel generation: Create new strategies from specs
    3. Explanation: Document strategy logic for review
    4. Hypothesis: Generate causal explanations

    IMPORTANT: All LLM operations are offline. Generated code is
    validated and compiled before any execution.
    """

    MUTATION_PROMPT = """You are a quantitative researcher improving a trading strategy.

Current strategy code:
```python
{strategy_code}
```

Backtest results:
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd:.1%}
- Profit Factor: {profit_factor:.2f}
- Win Rate: {win_rate:.1%}
- Trade Count: {trade_count}

Current market regime: {regime}

Modification request: {mutation_intent}

Analyze the strategy's weaknesses and improve it. Consider:
1. Signal quality - are the indicators appropriate?
2. Risk management - are stops/targets reasonable?
3. Regime adaptation - does it handle different market conditions?

Return ONLY the improved Python function. No explanations outside the code.
The function must be named 'evaluate' and take a 'features' numpy array parameter.

```python
def evaluate(features):
    '''Improved strategy: {mutation_intent}'''
    # Your improved code here
    return signal  # -1=sell, 0=hold, 1=buy
```"""

    NOVEL_PROMPT = """Generate a quantitative trading strategy with these specifications:

Target metrics:
- Sharpe Ratio: {target_sharpe:.1f}
- Max Drawdown: <{max_drawdown:.0%}
- Trades per month: ~{trades_per_month}

Market context:
- Regime: {regime} (bull/bear/range/volatile)
- Risk tolerance: {risk_tolerance} (conservative/moderate/aggressive)

Portfolio gaps (strategies we need that are different from existing):
{portfolio_gaps}

Available features (60-dimensional vector):
- Price features [0-14]: close, sma_20, sma_50, ema_12, ema_26, bb_upper, bb_lower, atr_14, etc.
- Volume features [15-24]: volume, obv, cvd, buy_volume_ratio
- Technical indicators [25-34]: rsi_14, macd, macd_signal, stoch_k, adx_14
- Order flow [35-44]: bid_ask_spread, order_book_imbalance, depth_imbalance
- Funding/carry [45-49]: funding_rate, funding_rate_zscore, open_interest, oi_change_1h
- Sentiment [50-54]: fear_greed_index, social_sentiment, btc_dominance
- Regime [55-59]: regime_label, regime_confidence, volatility_regime, trend_strength

Return JSON with this structure:
{{
    "strategy_name": "descriptive name",
    "entry_logic": "human readable entry conditions",
    "exit_logic": "human readable exit conditions",
    "causal_hypothesis": "WHY this strategy works mechanistically (200+ chars)",
    "stop_loss_atr": 2.0,
    "take_profit_atr": 3.0,
    "python_code": "def evaluate(features):\\n    ..."
}}"""

    EXPLANATION_PROMPT = """Analyze this trading strategy and explain:

```python
{strategy_code}
```

Backtest results:
- Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1%}
- Profit Factor: {profit_factor:.2f}, Win Rate: {win_rate:.1%}

Provide:
1. Plain English explanation of the logic
2. Market conditions where this works best
3. Risks and failure modes
4. Suggested improvements

Format as markdown."""

    def __init__(
        self,
        primary_client: LLMClient,
        fallback_client: Optional[LLMClient] = None,
        max_retries: int = 3
    ):
        """
        Args:
            primary_client: Main LLM client (e.g., Gemini)
            fallback_client: Backup client if primary fails
            max_retries: Retry attempts per request
        """
        self.primary = primary_client
        self.fallback = fallback_client
        self.max_retries = max_retries
        self.request_count = 0
        self.success_count = 0

    async def _generate_with_retry(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate with retry and fallback logic."""
        self.request_count += 1

        for attempt in range(self.max_retries):
            try:
                response = await self.primary.generate(prompt, max_tokens)
                if response:
                    self.success_count += 1
                    return response
            except Exception as e:
                logger.warning(f"Primary LLM attempt {attempt+1} failed: {e}")

        # Try fallback
        if self.fallback:
            try:
                response = await self.fallback.generate(prompt, max_tokens)
                if response:
                    self.success_count += 1
                    return response
            except Exception as e:
                logger.error(f"Fallback LLM failed: {e}")

        return ""

    async def mutate_strategy(
        self,
        strategy: StrategyGenome,
        backtest_result: Dict[str, Any],
        mutation_intent: str,
        regime: str = "normal"
    ) -> Optional[LLMGeneratedStrategy]:
        """
        Apply LLM-guided targeted mutation.

        Args:
            strategy: Strategy to mutate
            backtest_result: Dict with sharpe, max_drawdown, etc.
            mutation_intent: What kind of improvement to make
            regime: Current market regime

        Returns:
            LLMGeneratedStrategy or None if generation failed
        """
        prompt = self.MUTATION_PROMPT.format(
            strategy_code=strategy.to_python_code(),
            sharpe=backtest_result.get('sharpe', 0),
            max_dd=backtest_result.get('max_drawdown', 0),
            profit_factor=backtest_result.get('profit_factor', 1),
            win_rate=backtest_result.get('win_rate', 0),
            trade_count=backtest_result.get('trade_count', 0),
            regime=regime,
            mutation_intent=mutation_intent
        )

        response = await self._generate_with_retry(prompt)
        if not response:
            return None

        code = self._extract_code(response)
        if not code or not self._validate_syntax(code):
            return None

        return LLMGeneratedStrategy(
            code=code,
            explanation=f"LLM mutation: {mutation_intent}",
            causal_hypothesis="See mutation intent",
            confidence=0.6,
            raw_response=response
        )

    async def generate_novel(
        self,
        condition: GenerationCondition,
        portfolio_gaps: List[str],
        existing_strategies: Optional[List[str]] = None
    ) -> Optional[LLMGeneratedStrategy]:
        """
        Generate completely novel strategy from specifications.

        Args:
            condition: Target properties
            portfolio_gaps: List of strategy types needed
            existing_strategies: Names of strategies we already have

        Returns:
            LLMGeneratedStrategy or None
        """
        regime_map = {0: "bull", 1: "bear", 2: "range", 3: "volatile"}
        risk_map = {0: "conservative", 0.5: "moderate", 1: "aggressive"}

        gaps_text = "\n".join(f"- {g}" for g in portfolio_gaps)

        prompt = self.NOVEL_PROMPT.format(
            target_sharpe=condition.target_sharpe,
            max_drawdown=condition.target_max_drawdown,
            trades_per_month=condition.target_trades_per_month,
            regime=regime_map.get(condition.regime_label, "normal"),
            risk_tolerance=risk_map.get(round(condition.risk_tolerance), "moderate"),
            portfolio_gaps=gaps_text or "- General alpha generation"
        )

        response = await self._generate_with_retry(prompt, max_tokens=3000)
        if not response:
            return None

        result = self._parse_json(response)
        if not result:
            return None

        code = result.get("python_code", "")
        if not code or not self._validate_syntax(code):
            return None

        return LLMGeneratedStrategy(
            code=code,
            explanation=result.get("strategy_name", "LLM Generated"),
            causal_hypothesis=result.get("causal_hypothesis", ""),
            confidence=0.5,
            raw_response=response
        )

    async def explain_strategy(
        self,
        strategy: StrategyGenome,
        backtest_result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of strategy.

        Useful for review and documentation.
        """
        prompt = self.EXPLANATION_PROMPT.format(
            strategy_code=strategy.to_python_code(),
            sharpe=backtest_result.get('sharpe', 0),
            max_dd=backtest_result.get('max_drawdown', 0),
            profit_factor=backtest_result.get('profit_factor', 1),
            win_rate=backtest_result.get('win_rate', 0)
        )

        return await self._generate_with_retry(prompt, max_tokens=1500)

    async def generate_mutations(
        self,
        strategy: StrategyGenome,
        backtest_result: Dict[str, Any],
        num_mutations: int = 5
    ) -> List[LLMGeneratedStrategy]:
        """
        Generate multiple mutation variants.

        Uses different mutation intents for diversity.
        """
        mutation_intents = [
            "Improve entry signal precision to reduce false positives",
            "Add better exit timing to capture more profit",
            "Adapt to trending markets with dynamic stops",
            "Add regime-aware logic to handle volatility changes",
            "Simplify logic while maintaining performance",
            "Add order flow confirmation to entries",
            "Improve risk/reward ratio",
            "Add momentum confirmation",
            "Handle mean-reversion better",
            "Add funding rate consideration"
        ]

        selected_intents = mutation_intents[:num_mutations]

        tasks = [
            self.mutate_strategy(strategy, backtest_result, intent)
            for intent in selected_intents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results
                if isinstance(r, LLMGeneratedStrategy)]

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code block
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find function definition
        if "def evaluate" in text:
            start = text.find("def evaluate")
            # Find end by looking for next top-level definition or end
            lines = text[start:].split('\n')
            code_lines = []
            in_function = False

            for line in lines:
                if line.strip().startswith("def evaluate"):
                    in_function = True
                if in_function:
                    if line and not line[0].isspace() and not line.startswith("def evaluate"):
                        break
                    code_lines.append(line)

            return '\n'.join(code_lines)

        return text.strip()

    def _validate_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from response."""
        try:
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def strategy_from_llm(
        self,
        llm_strategy: LLMGeneratedStrategy
    ) -> Optional[StrategyGenome]:
        """
        Convert LLM-generated strategy to StrategyGenome.

        Compiles the code and wraps it in a genome structure.
        """
        try:
            # Compile code
            compiled = compile(llm_strategy.code, '<string>', 'exec')
            namespace = {}
            exec(compiled, namespace)

            if 'evaluate' not in namespace:
                return None

            # Create a genome that wraps the compiled function
            # This is a simplified conversion - real implementation
            # would parse the code structure into decision tree
            from ..core.genome import generate_random_strategy
            genome = generate_random_strategy(max_depth=3)
            genome.source_engine = "llm_guided"

            # Store the code for reference
            genome.backtest_metrics['llm_code'] = llm_strategy.code
            genome.backtest_metrics['causal_hypothesis'] = llm_strategy.causal_hypothesis

            return genome

        except Exception as e:
            logger.error(f"Failed to convert LLM strategy: {e}")
            return None


def create_llm_client(
    provider: str,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: One of 'ollama', 'gemini', 'deepseek'
        **kwargs: Provider-specific arguments

    Returns:
        LLMClient instance
    """
    if provider == 'ollama':
        return OllamaClient(
            base_url=kwargs.get('base_url', 'http://localhost:11434'),
            model=kwargs.get('model', 'deepseek-coder:33b')
        )
    elif provider == 'gemini':
        return GeminiClient(
            api_key=kwargs['api_key'],
            model=kwargs.get('model', 'gemini-1.5-pro')
        )
    elif provider == 'deepseek':
        return DeepSeekClient(
            api_key=kwargs['api_key'],
            base_url=kwargs.get('base_url', 'https://api.deepseek.com'),
            model=kwargs.get('model', 'deepseek-chat')
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
