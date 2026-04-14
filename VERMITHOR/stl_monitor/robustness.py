"""
Robustness Computation for Hybrid Dynamics.

Two distinct approaches are used, matching the nature of each signal:

Physical signals (thermal, battery)
────────────────────────────────────
  Thermal dynamics are inherently smooth due to heat capacity.
  We exploit this Lipschitz property to compute *direct* robustness:

      ρ(φ_thermal) = T_lim − T_t

  and a *safe horizon* — the minimum time until a violation is possible:

      Δt_safe = ρ / L_max

  where L_max is the maximum temperature rate of change (°C/s).

Network signals (latency, bandwidth)
─────────────────────────────────────
  Network conditions can change discontinuously (tunnels, congestion).
  Instead of Lipschitz bounds, we use Robust Conformal Prediction intervals
  [l_t, u_t] and evaluate *worst-case* robustness over the interval:

      ρ̄(φ_bw,  Γ_t) = inf_{b ∈ Γ_t} ρ(φ_bw,  b) = l_t − BW_min
      ρ̄(φ_lat, Γ_t) = inf_{b ∈ Γ_t} ρ(φ_lat, b) = RTT_max − u_t

This hybrid combination is a core novelty of the Thermo-Logical Inference framework.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from collections import deque


# =============================================================================
# Thermal robustness (Lipschitz-bounded)
# =============================================================================

@dataclass
class ThermalConfig:
    """Configuration for thermal robustness computation."""
    temp_limit: float       = 85.0   # °C — critical safety limit
    temp_warning: float     = 80.0   # °C — soft warning threshold
    recovery_target: float  = 75.0   # °C — target after a thermal event
    lipschitz_max: float    = 2.0    # °C/s — worst-case rate of change
    control_interval: float = 0.1   # s — control loop period


class ThermalRobustness:
    """
    Lipschitz-bounded robustness for device thermal dynamics.

    Because heat capacity bounds the rate of temperature change,
    we can provide a *guaranteed* time horizon during which the safety
    specification cannot be violated — even in the worst case.

    Specification:  φ_thermal = □[0,∞) (T < T_lim)
    Robustness:     ρ(φ_thermal, T) = T_lim − T
    Safe horizon:   Δt_safe = ρ / L_max
    """

    def __init__(self, config: ThermalConfig):
        self.config = config
        self.temperature_history: deque = deque(maxlen=100)
        self.time_history:        deque = deque(maxlen=100)

    def update(self, temperature: float, timestamp: float):
        """Record a new temperature measurement."""
        self.temperature_history.append(temperature)
        self.time_history.append(timestamp)

    def compute_robustness(self, temperature: float) -> float:
        """
        Compute quantitative robustness for the thermal safety specification.

            ρ > 0  →  safe (margin = ρ degrees)
            ρ ≤ 0  →  violation (spec already broken)
        """
        return self.config.temp_limit - temperature

    def compute_rate_of_change(self) -> float:
        """Estimate the current temperature rate of change in °C/s."""
        if len(self.temperature_history) < 2:
            return 0.0

        temps = list(self.temperature_history)[-5:]
        times = list(self.time_history)[-5:]
        dt    = times[-1] - times[0]

        if len(temps) < 2 or dt == 0.0:
            return 0.0

        return (temps[-1] - temps[0]) / dt

    def compute_safe_horizon(self, temperature: float) -> float:
        """
        Minimum time until a thermal violation could occur.

        Computed via the Lipschitz bound:  Δt = ρ / L_max.
        Returns 0 if already in violation.
        """
        rho = self.compute_robustness(temperature)
        return max(0.0, rho / self.config.lipschitz_max)

    def should_throttle(self, temperature: float) -> Tuple[bool, str]:
        """
        Determine whether thermal throttling is required.

        Returns:
            should_throttle: True if workload should be reduced
            reason:          Human-readable explanation
        """
        rho    = self.compute_robustness(temperature)
        rate   = self.compute_rate_of_change()
        horizon = self.compute_safe_horizon(temperature)

        if rho < 0:
            return True, (f"Thermal violation: T={temperature}°C "
                          f"> T_lim={self.config.temp_limit}°C")

        if temperature > self.config.temp_warning and rate > 0:
            return True, (f"Approaching limit: T={temperature}°C, "
                          f"rate={rate:.1f}°C/s")

        if horizon < self.config.control_interval * 3:
            return True, f"Insufficient safety margin: horizon={horizon:.2f}s"

        return False, f"Safe: ρ={rho:.1f}°C, horizon={horizon:.1f}s"


# =============================================================================
# Network robustness (worst-case over prediction intervals)
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration for network robustness computation."""
    min_bandwidth: float  = 1.0    # Mbps — minimum required for offloading
    max_latency: float    = 100.0  # ms  — maximum acceptable round-trip time
    history_window: int   = 50     # samples kept for rolling statistics


class NetworkRobustness:
    """
    Worst-case robustness for stochastic network dynamics.

    Unlike thermal dynamics, network conditions can change discontinuously
    (WiFi handoff, congestion bursts).  We therefore use Robust Conformal
    Prediction intervals [l_t, u_t] and evaluate safety with the conservative
    worst-case semantics:

        ρ̄(φ_bw,  Γ_t) = l_t − BW_min   (worst case: lower bound of BW interval)
        ρ̄(φ_lat, Γ_t) = RTT_max − u_t  (worst case: upper bound of RTT interval)

    Offloading is considered safe only when *both* worst-case robustness
    values are non-negative.
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.bandwidth_history: deque = deque(maxlen=config.history_window)
        self.latency_history:   deque = deque(maxlen=config.history_window)

    def update(self, bandwidth: float, latency: float):
        """Record a new network measurement."""
        self.bandwidth_history.append(bandwidth)
        self.latency_history.append(latency)

    def compute_robustness(
        self,
        bandwidth: float,
        latency: float,
    ) -> Tuple[float, float]:
        """
        Compute point robustness for bandwidth and latency constraints.

            φ_bw  = (BW > BW_min)    →  ρ_bw  = BW − BW_min
            φ_lat = (RTT < RTT_max)  →  ρ_lat = RTT_max − RTT
        """
        return (bandwidth - self.config.min_bandwidth,
                self.config.max_latency - latency)

    def compute_worst_case_robustness(
        self,
        bandwidth_interval: Tuple[float, float],
        latency_interval: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Compute worst-case robustness over robust CP prediction intervals.

            ρ̄_φ(Γ_t) = inf_{b ∈ Γ_t} ρ_φ(b)

        Bandwidth: worst case uses the *lower* bound of the CP interval.
        Latency:   worst case uses the *upper* bound of the CP interval.
        """
        bw_lower, _  = bandwidth_interval
        _, lat_upper = latency_interval

        return (bw_lower - self.config.min_bandwidth,
                self.config.max_latency - lat_upper)

    def can_offload(
        self,
        bandwidth_interval: Tuple[float, float],
        latency_interval: Tuple[float, float],
    ) -> Tuple[bool, str]:
        """
        Decide whether offloading is safe given current network uncertainty.

        Uses the conservative worst-case criterion: offload only when the
        worst-case robustness for both bandwidth and latency is non-negative.
        """
        rho_bw, rho_lat = self.compute_worst_case_robustness(
            bandwidth_interval, latency_interval
        )

        if rho_bw < 0:
            bw_lower, _ = bandwidth_interval
            return False, (f"Bandwidth too low: worst-case {bw_lower:.1f} Mbps "
                           f"< {self.config.min_bandwidth} Mbps")

        if rho_lat < 0:
            _, lat_upper = latency_interval
            return False, (f"Latency too high: worst-case {lat_upper:.1f} ms "
                           f"> {self.config.max_latency} ms")

        return True, f"Offload safe: ρ_bw={rho_bw:.1f}, ρ_lat={rho_lat:.1f}"

    def get_network_statistics(self) -> dict:
        """Return rolling mean and std for bandwidth and latency."""
        if not self.bandwidth_history:
            return {"bandwidth_mean": 0.0, "latency_mean": 0.0}

        return {
            "bandwidth_mean": float(np.mean(self.bandwidth_history)),
            "bandwidth_std":  float(np.std(self.bandwidth_history)),
            "latency_mean":   float(np.mean(self.latency_history)),
            "latency_std":    float(np.std(self.latency_history)),
        }
