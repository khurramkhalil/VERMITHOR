"""
f-Divergence Estimator for Distribution Shift Detection.

Key insight: A binary classifier trained to distinguish calibration data
(label=0) from runtime data (label=1) implicitly learns the likelihood ratio

    r(x) = p_test(x) / p_cal(x)

because the Bayes-optimal decision boundary satisfies

    p(y=1|x) / p(y=0|x)  =  p_test(x) / p_cal(x).

Given r(x), any f-divergence can be computed in O(1) via a single forward
pass, and an exponential moving average (EMA) provides a smooth online
estimate throughout inference.

Supported f-divergences
───────────────────────
  KL divergence   D_KL(P_test ∥ P_cal) = E_test[ log r(x) ]
  Chi-squared      D_χ²(P_test ∥ P_cal) = E_cal[ (r(x) − 1)² ]
  Total variation  D_TV(P_test ∥ P_cal) = 0.5 · E_cal[ |r(x) − 1| ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class DivergenceEstimatorConfig:
    """Configuration for the f-divergence estimator."""
    input_dim: int
    hidden_dim: int = 128
    f_divergence_type: str = "kl"   # "kl" | "chi_squared" | "tv"
    ema_decay: float = 0.99          # Smoothing factor for online EMA


class LikelihoodRatioEstimator(nn.Module):
    """
    Three-layer feedforward network for likelihood ratio estimation.

    Trained as a binary classifier (calibration=0, runtime=1).
    The network logits approximate log r(x) = log[p_test(x)/p_cal(x)].
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, D]

        Returns:
            logits: Log-likelihood ratio estimates [B, 1]
        """
        return self.net(x)

    def get_likelihood_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the likelihood ratio r(x) = p_test(x) / p_cal(x).

        Derived from classifier probabilities via:
            r(x) = p(y=1|x) / p(y=0|x)
        """
        logits = torch.clamp(self.forward(x), -10.0, 10.0)
        probs  = torch.sigmoid(logits)
        return probs / (1.0 - probs + 1e-8)


class FDivergenceEstimator(nn.Module):
    """
    Efficient f-divergence estimation for real-time distribution shift detection.

    The estimator is first trained on paired calibration / runtime batches
    (``train_on_calibration``).  Thereafter, ``estimate_divergence`` provides
    O(1) per-sample estimates and updates a bias-corrected EMA.

    Usage
    ─────
    estimator = FDivergenceEstimator(DivergenceEstimatorConfig(input_dim=64))
    estimator.train_on_calibration(cal_features, initial_runtime_features)

    # At every inference step:
    d = estimator.estimate_divergence(runtime_features)
    """

    def __init__(self, config: DivergenceEstimatorConfig):
        super().__init__()
        self.config = config

        self.ratio_estimator = LikelihoodRatioEstimator(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
        )

        # Running EMA statistics (non-trainable buffers)
        self.register_buffer("ema_divergence", torch.tensor(0.0))
        self.register_buffer("num_updates",    torch.tensor(0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_calibration(
        self,
        cal_data: torch.Tensor,
        runtime_data: torch.Tensor,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> float:
        """
        Train the likelihood ratio network to distinguish calibration from
        runtime data via binary cross-entropy.

        Args:
            cal_data:     Calibration features [N_cal, D]
            runtime_data: Runtime features     [N_run, D]
            epochs:       Number of training epochs
            lr:           Adam learning rate

        Returns:
            Final epoch loss
        """
        optimizer = torch.optim.Adam(self.ratio_estimator.parameters(), lr=lr)

        labels = torch.cat([
            torch.zeros(len(cal_data),     1),
            torch.ones( len(runtime_data), 1),
        ]).to(cal_data.device)
        data = torch.cat([cal_data, runtime_data], dim=0)

        final_loss = 0.0
        for _ in range(epochs):
            perm   = torch.randperm(len(data))
            logits = self.ratio_estimator(data[perm])
            loss   = F.binary_cross_entropy_with_logits(logits, labels[perm])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        return final_loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate_divergence(
        self,
        x: torch.Tensor,
        update_ema: bool = True,
    ) -> torch.Tensor:
        """
        Estimate the f-divergence for a batch of runtime samples.

        A single forward pass through the ratio network gives r(x);
        the divergence is then computed using the f-specific formula.
        The EMA is updated in-place.

        Args:
            x:          Runtime features [B, D]
            update_ema: Whether to update the running EMA

        Returns:
            Scalar divergence estimate
        """
        with torch.no_grad():
            ratios = self.ratio_estimator.get_likelihood_ratio(x)

            if self.config.f_divergence_type == "kl":
                # D_KL = E_test[ r · log r ]
                divergence = (ratios * torch.log(ratios + 1e-8)).mean()
            elif self.config.f_divergence_type == "chi_squared":
                # D_χ² = E_cal[ (r − 1)² ]
                divergence = ((ratios - 1.0) ** 2).mean()
            elif self.config.f_divergence_type == "tv":
                # D_TV = 0.5 · E_cal[ |r − 1| ]
                divergence = 0.5 * torch.abs(ratios - 1.0).mean()
            else:
                raise ValueError(
                    f"Unknown f-divergence type: {self.config.f_divergence_type!r}. "
                    f"Choose from 'kl', 'chi_squared', 'tv'."
                )

            if update_ema:
                self.num_updates += 1
                self.ema_divergence = (
                    self.config.ema_decay       * self.ema_divergence
                    + (1.0 - self.config.ema_decay) * divergence
                )

        return divergence

    def get_smoothed_divergence(self) -> torch.Tensor:
        """
        Return the bias-corrected EMA divergence estimate.

        Bias correction: d̂ = d_ema / (1 − decay^t)
        Prevents under-estimation early in the online update sequence.
        """
        if self.num_updates > 0:
            correction = 1.0 - self.config.ema_decay ** self.num_updates.float()
            return self.ema_divergence / correction
        return self.ema_divergence

    def reset_ema(self):
        """Reset the exponential moving average to zero."""
        self.ema_divergence.zero_()
        self.num_updates.zero_()
