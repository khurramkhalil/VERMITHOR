"""
Formal Hybrid Dynamics Model for Edge AI Orchestration.

Defines the mathematical framework distinguishing two signal classes:

  1. Lipschitz-bounded physical dynamics (temperature, battery)
     |y(t+Δt) − y(t)| ≤ L · Δt   for all t, Δt ≥ 0

     For these signals, robustness is computed *directly* from the current
     value without prediction intervals.

  2. Stochastic network dynamics (latency, bandwidth)
     Distribution can shift arbitrarily between calibration and runtime.
     Robustness is evaluated *worst-case* over a Robust Conformal Prediction
     interval inflated by the adaptive factor λ(D_f).

Key theorem (Separation): Physical and network robustness can be verified
independently and combined via conjunction semantics:

    ρ_hybrid = min(ρ_physical, ρ_network)

This separation makes real-time formal verification tractable.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class DynamicsType(Enum):
    """Classification of signal dynamics."""
    LIPSCHITZ_BOUNDED = "lipschitz"   # Thermal, battery
    STOCHASTIC        = "stochastic"  # Latency, bandwidth
    HYBRID            = "hybrid"      # Combined


# =============================================================================
# Dynamics specifications
# =============================================================================

@dataclass
class LipschitzDynamicsSpec:
    """
    Specification for a Lipschitz-bounded physical signal.

    Theorem: Given Lipschitz constant L and current value y(t),
    the signal cannot violate a threshold θ within Δt_safe = |θ − y| / L
    seconds, regardless of the future trajectory.
    """
    lipschitz_constant: float   # L (units per second)
    min_value: float
    max_value: float
    sampling_period_s: float = 0.1  # Control loop period Δt

    def max_change_per_step(self) -> float:
        """Maximum change within one control period: L · Δt."""
        return self.lipschitz_constant * self.sampling_period_s

    def safe_horizon(self, current_value: float, threshold: float) -> float:
        """
        Minimum guaranteed time until the threshold can be reached.

        Returns 0 if already at or past the threshold.
        """
        if current_value >= threshold:
            return 0.0
        return (threshold - current_value) / self.lipschitz_constant


@dataclass
class StochasticDynamicsSpec:
    """
    Specification for a stochastic network signal.

    Under bounded f-divergence D_f(P_test ∥ P_cal) ≤ D_max, the robust
    prediction region with inflation λ(D) maintains coverage ≥ 1 − α.

    Inflation function:
        λ(D) = min(λ_max,  1 + β · D)

    Robust interval (symmetric inflation around calibration quantile centre):
        Γ^{α,D}(x) = [c − w/2 · λ(D),  c + w/2 · λ(D)]
    where c, w are the centre and width of the calibration interval.
    """
    calibration_quantiles: Tuple[float, float]  # (lower, upper) from calibration data
    max_divergence: float                         # D_max: tolerated divergence bound
    inflation_base: float = 1.0                   # β: sensitivity to divergence
    inflation_max: float  = 3.0                   # λ_max: conservatism cap

    def compute_inflation(self, divergence: float) -> float:
        """Compute adaptive inflation λ(D)."""
        return min(self.inflation_max, 1.0 + self.inflation_base * divergence)

    def get_robust_interval(self, divergence: float) -> Tuple[float, float]:
        """Return the inflated prediction interval for the given divergence."""
        lam = self.compute_inflation(divergence)
        lower, upper = self.calibration_quantiles
        center     = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        return (center - half_width * lam, center + half_width * lam)


@dataclass
class HybridSystemSpec:
    """
    Complete specification of the hybrid edge AI system.

    The system has two classes of dynamics:
    - Physical (thermal, battery): Lipschitz-bounded  →  direct robustness
    - Network (latency, bandwidth): Stochastic         →  worst-case over CP intervals

    Key insight: These two classes can be verified independently and the
    overall system robustness is their minimum (conjunction semantics).
    """
    # Physical dynamics
    thermal: LipschitzDynamicsSpec
    battery: Optional[LipschitzDynamicsSpec] = None

    # Network dynamics
    latency:   Optional[StochasticDynamicsSpec] = None
    bandwidth: Optional[StochasticDynamicsSpec] = None

    # Safety thresholds
    temp_limit:     float = 85.0    # °C
    min_battery:    float = 10.0    # %
    max_latency:    float = 100.0   # ms
    min_bandwidth:  float = 10.0    # Mbps

    # Timing
    decision_period_s: float = 0.1  # Control loop period


# =============================================================================
# Robustness computation
# =============================================================================

class HybridRobustnessComputer:
    """
    Computes STL robustness for the hybrid system specification.

    Separation theorem: physical and network robustness are computed
    independently; their minimum gives the overall hybrid robustness.
    """

    def __init__(self, spec: HybridSystemSpec):
        self.spec = spec

    def compute_physical_robustness(
        self,
        temperature: float,
        battery: Optional[float] = None,
    ) -> float:
        """
        Compute robustness for Lipschitz-bounded physical signals.

        Direct computation (no approximation):
            ρ_T = T_lim − T        (temperature upper bound)
            ρ_B = B − B_min        (battery lower bound)
        """
        thermal_rob = self.spec.temp_limit - temperature

        if battery is not None and self.spec.battery is not None:
            battery_rob = battery - self.spec.min_battery
            return min(thermal_rob, battery_rob)

        return thermal_rob

    def compute_physical_horizon(
        self,
        temperature: float,
        battery: Optional[float] = None,
    ) -> float:
        """
        Compute the guaranteed safe time horizon for physical signals.

        Uses the Lipschitz bound to give a *hard* lower bound on the time
        until any safety violation can occur.
        """
        thermal_horizon = self.spec.thermal.safe_horizon(
            temperature, self.spec.temp_limit
        )

        if battery is not None and self.spec.battery is not None:
            remaining = self.spec.battery.max_value - battery
            battery_horizon = self.spec.battery.safe_horizon(
                remaining,
                self.spec.battery.max_value - self.spec.min_battery,
            )
            return min(thermal_horizon, battery_horizon)

        return thermal_horizon

    def compute_network_robustness(
        self,
        latency_interval: Tuple[float, float],
        bandwidth_interval: Tuple[float, float],
    ) -> float:
        """
        Compute worst-case robustness for stochastic network signals.

        For bandwidth:  worst-case = lower bound of CP interval
        For latency:    worst-case = upper bound of CP interval

        Returns the minimum (conjunction) of both.
        """
        _, lat_upper   = latency_interval
        bw_lower, _    = bandwidth_interval

        latency_rob   = self.spec.max_latency  - lat_upper
        bandwidth_rob = bw_lower - self.spec.min_bandwidth

        return min(latency_rob, bandwidth_rob)

    def compute_hybrid_robustness(
        self,
        temperature: float,
        latency_interval: Tuple[float, float],
        bandwidth_interval: Tuple[float, float],
        battery: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """
        Compute combined robustness for the full hybrid system.

        Main theorem:
            ρ_hybrid = min(ρ_physical, ρ_network)

        Both terms are computed independently, preserving the formal
        guarantees from their respective domains.

        Returns:
            overall_robustness: Scalar — min of physical and network
            breakdown:          Dict with per-component values
        """
        phys_rob = self.compute_physical_robustness(temperature, battery)
        net_rob  = self.compute_network_robustness(latency_interval, bandwidth_interval)
        hybrid   = min(phys_rob, net_rob)

        breakdown = {
            "physical":      phys_rob,
            "network":       net_rob,
            "hybrid":        hybrid,
            "thermal":       self.spec.temp_limit - temperature,
            "safe_horizon_s": self.compute_physical_horizon(temperature, battery),
        }

        return hybrid, breakdown


# =============================================================================
# Coverage theorem
# =============================================================================

class CoverageTheorem:
    """
    Implements the Robust Coverage Theorem for conformal prediction under shift.

    Theorem (Zhao et al. 2024):
    ───────────────────────────
    Let q_α = (1−α)-quantile of calibration nonconformity scores.
    Let D_f(P_test ∥ P_cal) ≤ D be the f-divergence bound.
    Define the inflation λ(D) = min(λ_max, 1 + β·D).

    For the robust prediction region
        Γ^{α,D}(x) = { y : s(x,y) ≤ λ(D) · q_α }

    it holds that:
        P_{(x,y)~P_test}(y ∈ Γ^{α,D}(x))  ≥  1 − α

    whenever the divergence bound holds.
    """

    def __init__(
        self,
        alpha: float       = 0.1,   # Target miscoverage rate
        beta: float        = 0.5,   # Inflation sensitivity
        lambda_max: float  = 3.0,   # Maximum inflation
        divergence_type: str = "kl",
    ):
        self.alpha          = alpha
        self.beta           = beta
        self.lambda_max     = lambda_max
        self.divergence_type = divergence_type

    def compute_required_inflation(self, divergence: float) -> float:
        """
        Compute λ(D) for a given divergence estimate.

        Grows with divergence (safety) while bounded by λ_max (anti-conservatism).
        """
        return min(self.lambda_max, 1.0 + self.beta * divergence)

    def verify_coverage_condition(
        self,
        empirical_coverage: float,
        divergence_estimate: float,
        tolerance: float = 0.01,
    ) -> Tuple[bool, str]:
        """
        Check whether empirical coverage meets the theoretical guarantee.

        Args:
            empirical_coverage:  Observed coverage fraction on test data
            divergence_estimate: Estimated D_f(P_test ∥ P_cal)
            tolerance:           Allowed slack in coverage check

        Returns:
            (satisfied, explanation)
        """
        target = 1.0 - self.alpha

        if empirical_coverage >= target - tolerance:
            return True, (f"Coverage {empirical_coverage:.3f} "
                          f"≥ target {target:.3f} ✓")
        return False, (f"Coverage {empirical_coverage:.3f} "
                       f"< target {target:.3f} ✗")

    def compute_optimal_beta(
        self,
        divergence_samples: List[float],
        coverage_samples: List[float],
    ) -> float:
        """
        Find the Pareto-optimal β that achieves 1−α coverage with minimum
        inflation (i.e. minimum interval width).

        Uses binary search over [0, 2] for 20 iterations.
        """
        beta_lo, beta_hi = 0.0, 2.0

        for _ in range(20):
            beta_mid = (beta_lo + beta_hi) / 2.0
            coverage = self._simulate_coverage(
                divergence_samples, coverage_samples, beta_mid
            )
            if coverage >= 1.0 - self.alpha:
                beta_hi = beta_mid   # Can afford smaller β
            else:
                beta_lo = beta_mid   # Need larger β

        return beta_hi  # Conservative choice

    def _simulate_coverage(
        self,
        divergences: List[float],
        base_coverages: List[float],
        beta: float,
    ) -> float:
        inflations = [min(self.lambda_max, 1.0 + beta * d) for d in divergences]
        adjusted   = [min(1.0, c * lam) for c, lam in zip(base_coverages, inflations)]
        return sum(adjusted) / len(adjusted)


# =============================================================================
# Factory
# =============================================================================

def create_default_hybrid_spec() -> HybridSystemSpec:
    """Return the default hybrid system specification for edge AI deployment."""
    return HybridSystemSpec(
        thermal=LipschitzDynamicsSpec(
            lipschitz_constant=5.0,   # 5°C/s max rise rate
            min_value=20.0,
            max_value=100.0,
            sampling_period_s=0.1,
        ),
        battery=LipschitzDynamicsSpec(
            lipschitz_constant=0.1,   # 0.1 %/s max drain
            min_value=0.0,
            max_value=100.0,
            sampling_period_s=0.1,
        ),
        latency=StochasticDynamicsSpec(
            calibration_quantiles=(5.0, 50.0),   # ms
            max_divergence=1.0,
            inflation_base=0.5,
            inflation_max=3.0,
        ),
        bandwidth=StochasticDynamicsSpec(
            calibration_quantiles=(20.0, 100.0),  # Mbps
            max_divergence=1.0,
            inflation_base=0.5,
            inflation_max=3.0,
        ),
        temp_limit=85.0,
        min_battery=10.0,
        max_latency=100.0,
        min_bandwidth=10.0,
    )
