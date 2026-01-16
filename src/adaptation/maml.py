"""
MAML Adaptation

When drift is detected, the strategy needs to adapt quickly—but not so
quickly that it forgets what worked before. Model-Agnostic Meta-Learning
(MAML) provides a solution: learn an initialization that adapts well to
new tasks with just a few gradient steps.

Key insight: Instead of training from scratch, MAML learns an
initialization that's close to optimal for many related tasks.
When drift occurs, 3-5 gradient steps adapt to the new regime.

Constraint: Only adapter layers update (base policy frozen).
This prevents catastrophic forgetting of core trading logic.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MAMLAdapter:
    """
    Rapid strategy adaptation using MAML.

    Only adapter layers update (base policy frozen).
    This prevents catastrophic forgetting of core trading logic.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_layers: List[str],
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ):
        """
        Args:
            base_model: Base neural network model
            adapter_layers: List of layer name prefixes to adapt
            inner_lr: Learning rate for inner loop adaptation
            inner_steps: Number of gradient steps for adaptation
        """
        self.base_model = base_model
        self.adapter_layers = adapter_layers
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # Freeze base, keep only adapters trainable
        self._freeze_base_layers()

    def _freeze_base_layers(self) -> None:
        """Freeze all layers except adapters."""
        for name, param in self.base_model.named_parameters():
            is_adapter = any(adapter in name for adapter in self.adapter_layers)
            param.requires_grad = is_adapter
            if not is_adapter:
                logger.debug(f"Frozen: {name}")

    def adapt(
        self,
        recent_data: torch.Tensor,
        recent_labels: torch.Tensor,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> Tuple[nn.Module, float]:
        """
        Adapt model to recent regime using few gradient steps.

        Args:
            recent_data: Recent feature data [N, feature_dim]
            recent_labels: Recent target labels [N]
            validation_data: Validation features [M, feature_dim]
            validation_labels: Validation targets [M]

        Returns:
            (adapted_model, validation_score)
        """
        # Deep copy for adaptation
        adapted_model = copy.deepcopy(self.base_model)

        # Get only adapter parameters
        adapter_params = [
            p for n, p in adapted_model.named_parameters()
            if any(adapter in n for adapter in self.adapter_layers) and p.requires_grad
        ]

        if not adapter_params:
            logger.warning("No adapter parameters found for adaptation")
            return adapted_model, 0.0

        # Inner loop optimization
        inner_opt = torch.optim.SGD(adapter_params, lr=self.inner_lr)

        adapted_model.train()
        for step in range(self.inner_steps):
            predictions = adapted_model(recent_data)
            loss = nn.MSELoss()(predictions.squeeze(), recent_labels)

            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

            logger.debug(f"Adaptation step {step+1}/{self.inner_steps}, loss: {loss.item():.4f}")

        # Evaluate on validation
        adapted_model.eval()
        with torch.no_grad():
            val_pred = adapted_model(validation_data)
            returns = val_pred.squeeze() * validation_labels
            val_score = (returns.mean() / (returns.std() + 1e-8)).item()

        return adapted_model, val_score


class MAMLMetaLearner:
    """
    Meta-learner for training MAML initialization.

    Trains the base model such that it can quickly adapt to new tasks
    (market regimes) with just a few gradient steps.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_layers: List[str],
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        meta_batch_size: int = 4
    ):
        """
        Args:
            model: Model to meta-train
            adapter_layers: Layers for inner loop adaptation
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            inner_steps: Steps per task adaptation
            meta_batch_size: Number of tasks per meta-update
        """
        self.model = model
        self.adapter_layers = adapter_layers
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size

        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        self.train_losses: List[float] = []

    def meta_train_step(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Single meta-training step.

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) for each task

        Returns:
            Meta-loss value
        """
        self.model.train()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in task_batch:
            # Create task-specific copy
            task_model = copy.deepcopy(self.model)

            # Inner loop: adapt to support set
            adapter_params = [
                p for n, p in task_model.named_parameters()
                if any(adapter in n for adapter in self.adapter_layers)
            ]

            for _ in range(self.inner_steps):
                support_pred = task_model(support_x)
                support_loss = nn.MSELoss()(support_pred.squeeze(), support_y)

                grads = torch.autograd.grad(support_loss, adapter_params)
                for param, grad in zip(adapter_params, grads):
                    param.data -= self.inner_lr * grad

            # Outer loop: evaluate on query set
            query_pred = task_model(query_x)
            task_loss = nn.MSELoss()(query_pred.squeeze(), query_y)
            meta_loss += task_loss

        # Meta-update
        meta_loss = meta_loss / len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()

        loss_val = meta_loss.item()
        self.train_losses.append(loss_val)
        return loss_val


class AdaptiveStrategyNetwork(nn.Module):
    """
    Neural network with explicit adapter layers for MAML.

    Architecture:
    - Base layers: Core feature extraction (frozen during adaptation)
    - Adapter layers: Regime-specific adjustments (updated during adaptation)
    - Output: Trading signal prediction
    """

    def __init__(
        self,
        input_dim: int = 60,
        hidden_dim: int = 128,
        adapter_dim: int = 32
    ):
        super().__init__()

        # Base layers (frozen during adaptation)
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Adapter layers (updated during adaptation)
        self.adapter_1 = nn.Linear(hidden_dim, adapter_dim)
        self.adapter_2 = nn.Linear(adapter_dim, adapter_dim)

        # Output layer
        self.output = nn.Linear(hidden_dim + adapter_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base features
        base_features = self.base(x)

        # Adapter features
        adapter_features = torch.relu(self.adapter_1(base_features))
        adapter_features = torch.relu(self.adapter_2(adapter_features))

        # Combine and output
        combined = torch.cat([base_features, adapter_features], dim=-1)
        return self.output(combined)

    def get_adapter_layer_names(self) -> List[str]:
        """Get names of adapter layers."""
        return ['adapter_1', 'adapter_2']


class RegimeAdaptationManager:
    """
    Manages strategy adaptation across market regimes.

    Maintains adapted models for each regime and switches between them.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_layers: List[str],
        regimes: List[str] = None
    ):
        """
        Args:
            base_model: Base model for adaptation
            adapter_layers: Layers to adapt
            regimes: List of regime names
        """
        self.base_model = base_model
        self.adapter_layers = adapter_layers
        self.regimes = regimes or ['bull', 'bear', 'range', 'volatile']

        self.maml_adapter = MAMLAdapter(base_model, adapter_layers)
        self.regime_models: Dict[str, nn.Module] = {}
        self.current_regime = 'range'

    def adapt_to_regime(
        self,
        regime: str,
        regime_data: torch.Tensor,
        regime_labels: torch.Tensor,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> float:
        """
        Adapt model to specific regime.

        Args:
            regime: Regime name
            regime_data: Training data for regime
            regime_labels: Training labels
            validation_data: Validation data
            validation_labels: Validation labels

        Returns:
            Validation score for adapted model
        """
        adapted_model, val_score = self.maml_adapter.adapt(
            regime_data, regime_labels,
            validation_data, validation_labels
        )

        self.regime_models[regime] = adapted_model
        logger.info(f"Adapted to regime '{regime}' with validation score: {val_score:.4f}")

        return val_score

    def switch_regime(self, regime: str) -> bool:
        """
        Switch to a different regime's adapted model.

        Args:
            regime: Target regime

        Returns:
            True if switch successful
        """
        if regime in self.regime_models:
            self.current_regime = regime
            logger.info(f"Switched to regime: {regime}")
            return True
        else:
            logger.warning(f"No adapted model for regime: {regime}")
            return False

    def get_current_model(self) -> nn.Module:
        """Get currently active model."""
        if self.current_regime in self.regime_models:
            return self.regime_models[self.current_regime]
        return self.base_model

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Make prediction using current regime model."""
        model = self.get_current_model()
        model.eval()
        with torch.no_grad():
            return model(features)
