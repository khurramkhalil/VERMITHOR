"""
Real-Time f-Divergence Estimation with O(1) Amortised Updates.

Key contributions of this module
──────────────────────────────────
1. O(1) amortised per-sample divergence estimation via EMA (no batch
   retraining required at every inference step).
2. Lightweight likelihood ratio network (2-layer + LayerNorm) suited for
   deployment on resource-constrained edge devices.
3. Memory-bounded calibration buffer (FIFO, configurable size).
4. PID-inspired AdaptiveInflationController that combines proportional,
   integral, and derivative terms for stable yet responsive λ(D) values.

Training procedure
───────────────────
The ratio network is trained to classify calibration (label=0) vs. runtime
(label=1) features via binary cross-entropy.  The resulting logits
approximate the log-likelihood ratio log r(x) = log[p_test(x)/p_cal(x)],
from which any f-divergence can be computed in one forward pass.

Optional ``fine_tune()`` allows periodic retraining when substantial shift
is detected, without disrupting the running EMA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnlineDivergenceConfig:
    """Configuration for online f-divergence estimation."""
    # Ratio network
    hidden_dim: int  = 64
    num_layers: int  = 2

    # EMA / online estimation
    ema_alpha: float     = 0.1    # Weight given to new observations
    warmup_samples: int  = 100    # Minimum calibration samples before estimating

    # Memory
    buffer_size: int = 1000  # Maximum calibration samples retained (FIFO)
    batch_size: int  = 32    # Batch size for ratio estimation

    # Divergence
    divergence_type: str = "kl"    # "kl" | "chi2" | "tv"
    clip_ratio: float    = 10.0    # Clip likelihood ratios for stability


# =============================================================================
# Likelihood Ratio Network
# =============================================================================

class LikelihoodRatioNetwork(nn.Module):
    """
    Lightweight feedforward network for likelihood ratio estimation.

    The Bayes-optimal classifier between P_cal and P_test satisfies:

        p(y=1|x) / p(y=0|x)  =  p_test(x) / p_cal(x)  =  r(x)

    This network is trained as that classifier; its output logits are
    therefore proportional to log r(x).

    Architecture (per layer): Linear → LayerNorm → ReLU → Dropout
    Final layer: Linear → scalar log-ratio
    """

    def __init__(self, input_dim: int, config: OnlineDivergenceConfig):
        super().__init__()
        layers    = []
        curr_dim  = input_dim

        for _ in range(config.num_layers):
            layers += [
                nn.Linear(curr_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
            curr_dim = config.hidden_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.network   = nn.Sequential(*layers)
        self.clip_ratio = config.clip_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood ratio log r(x) for each sample.

        Args:
            x: Input features [B, D]

        Returns:
            log_ratio: [B]
        """
        return self.network(x).squeeze(-1)

    def get_likelihood_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute r(x) = p_test(x) / p_cal(x) with numerical clipping.
        """
        log_ratio = self.forward(x)
        ratio     = torch.exp(log_ratio)
        return torch.clamp(ratio, 0.0, self.clip_ratio)


# =============================================================================
# Online f-Divergence Estimator
# =============================================================================

class OnlineFDivergenceEstimator(nn.Module):
    """
    Real-time f-divergence estimation with O(1) amortised updates.

    Algorithm
    ──────────
    1. Maintain a fixed-size FIFO buffer of calibration samples.
    2. On each call to ``estimate_divergence``, perform one forward pass
       through the ratio network to obtain r(x).
    3. Compute the f-divergence estimate from r(x) using the selected formula.
    4. Update the EMA and variance trackers.

    Complexity
    ──────────
    - Inference: O(1) per sample (single forward pass, no optimisation step).
    - Fine-tuning: O(B · steps) on demand, where B = batch_size.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[OnlineDivergenceConfig] = None,
    ):
        super().__init__()
        self.config    = config or OnlineDivergenceConfig()
        self.input_dim = input_dim

        self.ratio_net = LikelihoodRatioNetwork(input_dim, self.config)

        # Calibration buffer (FIFO)
        self.cal_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Online statistics (non-trainable)
        self.register_buffer("ema_divergence", torch.tensor(0.0))
        self.register_buffer("ema_variance",   torch.tensor(1.0))
        self.register_buffer("num_updates",    torch.tensor(0))

        self.is_calibrated = False

    def add_calibration_data(self, data: torch.Tensor):
        """
        Add samples from the calibration distribution to the FIFO buffer.

        Once the buffer holds ≥ ``warmup_samples`` entries, the estimator
        is marked as calibrated and divergence estimates become available.

        Args:
            data: Calibration samples [N, D]
        """
        for sample in data:
            self.cal_buffer.append(sample.detach().cpu())

        if len(self.cal_buffer) >= self.config.warmup_samples:
            self.is_calibrated = True

    def estimate_divergence(
        self,
        test_data: torch.Tensor,
        update_ema: bool = True,
    ) -> torch.Tensor:
        """
        Estimate f-divergence between calibration and test distributions.

        Returns 0 before the estimator is calibrated (during warm-up).

        Args:
            test_data:  Test (runtime) samples [N, D]
            update_ema: Whether to update the running EMA

        Returns:
            Scalar divergence estimate
        """
        if not self.is_calibrated:
            return torch.tensor(0.0, device=test_data.device)

        # Sample a calibration mini-batch
        idx       = torch.randperm(len(self.cal_buffer))[: self.config.batch_size]
        cal_batch = torch.stack([self.cal_buffer[i] for i in idx]).to(test_data.device)

        if self.config.divergence_type == "kl":
            div = self._estimate_kl(cal_batch, test_data)
        elif self.config.divergence_type == "chi2":
            div = self._estimate_chi2(cal_batch, test_data)
        elif self.config.divergence_type == "tv":
            div = self._estimate_tv(cal_batch, test_data)
        else:
            div = self._estimate_kl(cal_batch, test_data)

        if update_ema:
            alpha = self.config.ema_alpha
            self.ema_divergence = (1.0 - alpha) * self.ema_divergence + alpha * div.detach()
            delta               = div.detach() - self.ema_divergence
            self.ema_variance   = (1.0 - alpha) * self.ema_variance   + alpha * delta ** 2
            self.num_updates   += 1

        return div

    # ------------------------------------------------------------------
    # f-Divergence estimators
    # ------------------------------------------------------------------

    def _estimate_kl(self, cal: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """
        D_KL(P_test ∥ P_cal) = E_{x~P_test}[ log r(x) ]
        """
        with torch.no_grad():
            log_ratios = self.ratio_net(test)
            return torch.clamp(log_ratios.mean(), min=0.0)

    def _estimate_chi2(self, cal: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """
        D_χ²(P_test ∥ P_cal) = E_{x~P_cal}[ (r(x) − 1)² ]
        """
        with torch.no_grad():
            ratios = self.ratio_net.get_likelihood_ratio(cal)
            return ((ratios - 1.0) ** 2).mean()

    def _estimate_tv(self, cal: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """
        D_TV(P_test ∥ P_cal) = 0.5 · E_{x~P_cal}[ |r(x) − 1| ]
        """
        with torch.no_grad():
            ratios = self.ratio_net.get_likelihood_ratio(cal)
            return 0.5 * torch.abs(ratios - 1.0).mean()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_divergence_with_uncertainty(self) -> Tuple[float, float]:
        """
        Return (mean, std) of the running divergence estimate.

        std is the square root of the EMA variance tracker.
        """
        return self.ema_divergence.item(), math.sqrt(self.ema_variance.item())

    # ------------------------------------------------------------------
    # Optional fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        test_data: torch.Tensor,
        num_steps: int = 10,
        lr: float = 1e-4,
    ):
        """
        Periodically retrain the ratio network on fresh test data.

        Should be called when a large divergence estimate signals that the
        ratio network may have drifted from the true likelihood ratio.
        Does not reset the EMA.
        """
        if not self.is_calibrated:
            return

        optimizer = torch.optim.Adam(self.ratio_net.parameters(), lr=lr)

        for _ in range(num_steps):
            cal_idx   = torch.randperm(len(self.cal_buffer))[: self.config.batch_size]
            cal_batch = torch.stack([self.cal_buffer[i] for i in cal_idx]).to(test_data.device)

            test_idx  = torch.randperm(len(test_data))[: self.config.batch_size]
            test_batch = test_data[test_idx]

            cal_loss  = F.binary_cross_entropy_with_logits(
                self.ratio_net(cal_batch), torch.zeros(len(cal_batch), device=cal_batch.device)
            )
            test_loss = F.binary_cross_entropy_with_logits(
                self.ratio_net(test_batch), torch.ones(len(test_batch), device=test_batch.device)
            )

            optimizer.zero_grad()
            (cal_loss + test_loss).backward()
            optimizer.step()

    def reset(self):
        """Reset all state including buffer, EMA, and calibration flag."""
        self.cal_buffer.clear()
        self.ema_divergence.zero_()
        self.ema_variance.fill_(1.0)
        self.num_updates.zero_()
        self.is_calibrated = False


# =============================================================================
# Adaptive Inflation Controller (PID-inspired)
# =============================================================================

class AdaptiveInflationController:
    """
    PID-inspired controller for adaptive prediction interval inflation.

    Three terms contribute to the inflation factor λ:
      Proportional: responds immediately to the current divergence.
      Integral:     accumulates persistent shift over a rolling window.
      Derivative:   damps rapid oscillation in λ.

    Final inflation:
        λ = clip(λ_base + P + I − D,  λ_base,  λ_max)
    """

    def __init__(
        self,
        lambda_base: float   = 1.0,
        lambda_max: float    = 3.0,
        beta: float          = 0.5,    # Proportional gain
        integral_window: int = 10,     # Window length for integral term
        damping: float       = 0.1,    # Derivative damping coefficient
    ):
        self.lambda_base     = lambda_base
        self.lambda_max      = lambda_max
        self.beta            = beta
        self.integral_window = integral_window
        self.damping         = damping

        self.divergence_history: deque = deque(maxlen=integral_window)
        self.prev_lambda = lambda_base

    def compute_inflation(self, divergence: float) -> float:
        """
        Compute the adaptive inflation factor for the given divergence estimate.

        Args:
            divergence: Current f-divergence estimate

        Returns:
            λ ∈ [λ_base, λ_max]
        """
        # Proportional term
        prop = self.beta * divergence

        # Integral term — average divergence over the rolling window
        self.divergence_history.append(divergence)
        integral  = sum(self.divergence_history) / len(self.divergence_history)
        int_term  = 0.5 * self.beta * integral

        # Derivative term — damping based on recent λ change
        deriv_term = self.damping * (self.prev_lambda - self.lambda_base)

        raw_lambda  = self.lambda_base + prop + int_term - deriv_term
        new_lambda  = max(self.lambda_base, min(self.lambda_max, raw_lambda))
        self.prev_lambda = new_lambda
        return new_lambda

    def get_inflated_interval(
        self,
        interval: Tuple[float, float],
        divergence: float,
    ) -> Tuple[float, float]:
        """
        Apply adaptive inflation to a prediction interval.

        The interval is expanded symmetrically around its centre:
            [c − w/2 · λ,  c + w/2 · λ]

        Args:
            interval:   (lower, upper) bounds
            divergence: Current divergence estimate

        Returns:
            Inflated (lower, upper) bounds
        """
        lam        = self.compute_inflation(divergence)
        lower, upper = interval
        center     = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        return (center - half_width * lam, center + half_width * lam)

    def reset(self):
        """Reset controller history."""
        self.divergence_history.clear()
        self.prev_lambda = self.lambda_base
