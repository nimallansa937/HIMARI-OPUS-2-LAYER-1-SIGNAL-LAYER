"""
Surrogate Model for HIFA Stage 2

Neural network that predicts strategy performance (Sharpe ratio)
from the 127-dimensional strategy vector encoding.

Enables cheap filtering before expensive backtests.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model."""
    input_dim: int = 127
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    output_uncertainty: bool = True


class SurrogateModel(nn.Module):
    """
    Neural network for predicting strategy performance.

    Input: 127-dim strategy vector
    Output: (predicted_sharpe, uncertainty) if output_uncertainty else predicted_sharpe

    Architecture: MLP with residual connections and layer normalization.
    """

    def __init__(self, config: Optional[SurrogateConfig] = None):
        super().__init__()
        self.config = config or SurrogateConfig()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU()
        )

        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.config.dropout)
            ))

        # Output heads
        if self.config.output_uncertainty:
            # Predict mean and log variance (for uncertainty)
            self.mean_head = nn.Linear(self.config.hidden_dim, 1)
            self.logvar_head = nn.Linear(self.config.hidden_dim, 1)
        else:
            self.output_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Strategy vectors [batch, 127]

        Returns:
            If output_uncertainty: [batch, 2] (mean, std)
            Else: [batch, 1] (mean only)
        """
        h = self.input_proj(x)

        # Residual blocks
        for layer in self.layers:
            h = h + layer(h)

        if self.config.output_uncertainty:
            mean = self.mean_head(h)
            logvar = self.logvar_head(h)
            std = torch.exp(0.5 * logvar)
            return torch.cat([mean, std], dim=-1)
        else:
            return self.output_head(h)

    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Predict with uncertainty for a single strategy.

        Returns:
            (predicted_sharpe, uncertainty)
        """
        self.eval()
        with torch.no_grad():
            output = self(x.unsqueeze(0) if x.dim() == 1 else x)

        if self.config.output_uncertainty:
            return output[0, 0].item(), output[0, 1].item()
        else:
            return output[0, 0].item(), 0.5  # Default uncertainty


class SurrogateTrainer:
    """Training utilities for surrogate model."""

    def __init__(
        self,
        model: SurrogateModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_step(
        self,
        strategy_vectors: torch.Tensor,
        target_sharpes: torch.Tensor
    ) -> float:
        """
        Single training step.

        Args:
            strategy_vectors: [batch, 127]
            target_sharpes: [batch, 1]

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(strategy_vectors)

        if self.model.config.output_uncertainty:
            mean = output[:, 0:1]
            logvar = output[:, 1:2]
            # Gaussian NLL loss
            loss = 0.5 * (logvar + (target_sharpes - mean)**2 / torch.exp(logvar))
            loss = loss.mean()
        else:
            loss = nn.MSELoss()(output, target_sharpes)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.train_losses.append(loss_val)
        return loss_val

    def validate(
        self,
        strategy_vectors: torch.Tensor,
        target_sharpes: torch.Tensor
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(strategy_vectors)

            if self.model.config.output_uncertainty:
                mean = output[:, 0:1]
                loss = nn.MSELoss()(mean, target_sharpes)
            else:
                loss = nn.MSELoss()(output, target_sharpes)

        return loss.item()

    def train_epoch(
        self,
        train_loader,
        val_loader=None
    ) -> Tuple[float, Optional[float]]:
        """
        Train for one epoch.

        Returns:
            (train_loss, val_loss or None)
        """
        epoch_train_losses = []

        for vectors, targets in train_loader:
            loss = self.train_step(vectors, targets)
            epoch_train_losses.append(loss)

        train_loss = np.mean(epoch_train_losses)

        val_loss = None
        if val_loader:
            val_losses = []
            for vectors, targets in val_loader:
                loss = self.validate(vectors, targets)
                val_losses.append(loss)
            val_loss = np.mean(val_losses)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

        return train_loss, val_loss

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


class EnsembleSurrogate(nn.Module):
    """
    Ensemble of surrogate models for better uncertainty estimation.

    Uses multiple models with different initializations.
    Uncertainty = standard deviation of predictions across ensemble.
    """

    def __init__(
        self,
        n_models: int = 5,
        config: Optional[SurrogateConfig] = None
    ):
        super().__init__()
        self.n_models = n_models

        # Force uncertainty off for individual models (we compute it from ensemble)
        config = config or SurrogateConfig()
        config.output_uncertainty = False

        self.models = nn.ModuleList([
            SurrogateModel(config) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble forward pass.

        Returns:
            [batch, 2] with (mean_prediction, uncertainty)
        """
        predictions = torch.stack([model(x) for model in self.models], dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return torch.cat([mean, std], dim=-1)

    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """Predict with uncertainty for single input."""
        self.eval()
        with torch.no_grad():
            output = self(x.unsqueeze(0) if x.dim() == 1 else x)
        return output[0, 0].item(), output[0, 1].item()


class SurrogateDataset(torch.utils.data.Dataset):
    """Dataset for surrogate model training."""

    def __init__(
        self,
        strategy_vectors: np.ndarray,
        sharpe_values: np.ndarray
    ):
        """
        Args:
            strategy_vectors: [N, 127] array of strategy encodings
            sharpe_values: [N] array of corresponding Sharpe ratios
        """
        self.vectors = torch.tensor(strategy_vectors, dtype=torch.float32)
        self.sharpes = torch.tensor(sharpe_values, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vectors[idx], self.sharpes[idx]


def create_surrogate_dataloader(
    strategies: List,
    batch_size: int = 32,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create dataloader from list of strategies with backtest results.

    Args:
        strategies: List of StrategyGenome with backtest_metrics
        batch_size: Batch size for training
        shuffle: Whether to shuffle data

    Returns:
        DataLoader for surrogate training
    """
    vectors = np.stack([s.to_vector() for s in strategies])
    sharpes = np.array([s.backtest_metrics.get('sharpe', 0) for s in strategies])

    dataset = SurrogateDataset(vectors, sharpes)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
