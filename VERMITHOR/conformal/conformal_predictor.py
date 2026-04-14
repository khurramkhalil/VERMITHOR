"""
Conformal Prediction for Uncertainty Quantification.

Implements two predictors:

1. ``ConformalPredictor`` — standard split conformal prediction providing
   distribution-free coverage guarantees under exchangeability.

2. ``RobustConformalPredictor`` — extends CP to settings with distribution
   shift, following Zhao et al. "Robust Conformal Prediction for STL Runtime
   Verification under Distribution Shift" (2024).

   The key contribution is an adaptive inflation function

       λ(D_f) = min(λ_max,  λ_base × (1 + β × D_f))

   that widens prediction intervals proportionally to the estimated
   f-divergence between calibration and runtime distributions.
   This maintains coverage ≥ 1 − α whenever D_f ≤ D_max.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from .divergence_estimator import FDivergenceEstimator, DivergenceEstimatorConfig


# =============================================================================
# Standard Conformal Predictor
# =============================================================================

@dataclass
class ConformalConfig:
    """Configuration for the standard conformal predictor."""
    coverage_probability: float = 0.9   # Target coverage 1 − α
    calibration_size: int = 500         # Expected calibration-set size


class ConformalPredictor:
    """
    Standard split conformal prediction for regression / prediction intervals.

    Calibration procedure
    ─────────────────────
    Given a hold-out calibration set {(x_i, y_i)}_{i=1}^n:
      1. Compute nonconformity scores  s_i = ‖ŷ_i − y_i‖.
      2. Sort scores and compute the ⌈(n+1)(1−α)⌉/n quantile → q̂.

    Prediction
    ──────────
    For a new point x: return the symmetric interval [ŷ − q̂,  ŷ + q̂].

    Under exchangeability this guarantees P(y ∈ Ĉ(x)) ≥ 1 − α.
    """

    def __init__(self, config: ConformalConfig):
        self.config = config
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None

    def calibrate(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Fit the conformal quantile on held-out calibration data.

        Args:
            predictions:  Model predictions  [N, D]
            ground_truth: True values         [N, D]
        """
        scores = np.abs(predictions - ground_truth)
        if scores.ndim > 1:
            scores = np.linalg.norm(scores, axis=1)

        self.calibration_scores = np.sort(scores)
        n       = len(self.calibration_scores)
        q_level = min(1.0, np.ceil((n + 1) * self.config.coverage_probability) / n)
        self.quantile = float(np.quantile(self.calibration_scores, q_level))

    def predict(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate symmetric prediction intervals.

        Args:
            predictions: Point predictions [N, D]

        Returns:
            lower: Lower bounds  [N, D]
            upper: Upper bounds  [N, D]
        """
        if self.quantile is None:
            raise RuntimeError("Predictor must be calibrated before calling predict().")
        return predictions - self.quantile, predictions + self.quantile

    def get_interval_width(self) -> float:
        """Return the total width of the current prediction interval."""
        return float("inf") if self.quantile is None else 2.0 * self.quantile


# =============================================================================
# Robust Conformal Predictor
# =============================================================================

@dataclass
class RobustConformalConfig:
    """Configuration for the robust conformal predictor."""
    base_coverage: float  = 0.9    # Target 1 − α
    lambda_base: float    = 1.0    # Minimum inflation factor
    lambda_max: float     = 3.0    # Maximum inflation (prevents degenerate intervals)
    beta: float           = 1.0    # Sensitivity of inflation to divergence
    divergence_type: str  = "kl"   # f-divergence type: "kl" | "chi_squared" | "tv"
    feature_dim: int      = 64     # Feature dimension for divergence estimator


class RobustConformalPredictor:
    """
    Robust Conformal Prediction under distribution shift.

    The inflation function

        λ(D_f) = min(λ_max,  λ_base × (1 + β · D_f))

    adaptively widens prediction intervals as the estimated f-divergence
    D_f(P_test ∥ P_cal) grows, providing formal coverage guarantees even when
    the runtime distribution deviates from the calibration distribution.

    Guarantee (Zhao et al. 2024):
        P_{(x,y)~P_test}(y ∈ Γ^{α,D}(x))  ≥  1 − α
    whenever D_f(P_test ∥ P_cal) ≤ D_max.
    """

    def __init__(self, config: RobustConformalConfig):
        self.config = config

        self.base_predictor = ConformalPredictor(
            ConformalConfig(coverage_probability=config.base_coverage)
        )

        div_config = DivergenceEstimatorConfig(
            input_dim=config.feature_dim,
            f_divergence_type=config.divergence_type,
        )
        self.divergence_estimator = FDivergenceEstimator(div_config)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        calibration_features: torch.Tensor,
    ):
        """
        Calibrate the base predictor and store calibration features for
        divergence estimation.

        Args:
            predictions:           Model predictions              [N, D]
            ground_truth:          True values                   [N, D]
            calibration_features:  Network features at calibration [N, feature_dim]
        """
        self.base_predictor.calibrate(predictions, ground_truth)
        self._calibration_features = calibration_features

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------

    def compute_inflation_factor(self, runtime_features: torch.Tensor) -> float:
        """
        Compute the adaptive inflation factor λ(D_f).

        λ(D_f) = min(λ_max,  λ_base × (1 + β · D_f))

        Args:
            runtime_features: Current runtime features [B, feature_dim]

        Returns:
            Scalar inflation factor in [λ_base, λ_max]
        """
        D_f = self.divergence_estimator.estimate_divergence(runtime_features)
        D_f = float(D_f.item()) if isinstance(D_f, torch.Tensor) else float(D_f)
        D_f = max(0.0, min(D_f, 10.0))  # Clamp for numerical stability

        inflation = self.config.lambda_base * (1.0 + self.config.beta * D_f)
        return min(inflation, self.config.lambda_max)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        predictions: np.ndarray,
        runtime_features: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate robust prediction intervals.

        Args:
            predictions:      Point predictions [N, D]
            runtime_features: Current runtime features for divergence estimation

        Returns:
            lower:     Lower bounds        [N, D]
            upper:     Upper bounds        [N, D]
            inflation: Applied λ factor
        """
        inflation = (
            self.compute_inflation_factor(runtime_features)
            if runtime_features is not None
            else self.config.lambda_base
        )

        base_lower, base_upper = self.base_predictor.predict(predictions)
        center     = (base_lower + base_upper) / 2.0
        half_width = (base_upper - base_lower) / 2.0 * inflation

        return center - half_width, center + half_width, inflation

    def get_robust_prediction_region(
        self,
        predictions: np.ndarray,
        runtime_features: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute the robust prediction region Γ^{α,D}(x) for STL verification.

        Returns:
            lower:      Lower bounds
            upper:      Upper bounds
            inflation:  Applied λ factor
            divergence: Current smoothed D_f estimate
        """
        divergence = float(self.divergence_estimator.get_smoothed_divergence().item())
        lower, upper, inflation = self.predict(predictions, runtime_features)
        return lower, upper, inflation, divergence
