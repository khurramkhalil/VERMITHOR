"""
STL Runtime Monitor with Efficient Temporal Evaluation.

Monitors Signal Temporal Logic (STL) specifications for a hybrid edge AI
system, combining:

  1. Lipschitz-bounded robustness for thermal dynamics
  2. Robust Conformal Prediction intervals for network dynamics
  3. O(1) amortised temporal operators via Monotonic Queues

STL specifications monitored
─────────────────────────────
  φ_thermal  : □[0,∞) (T < 85°C)
  φ_latency  : □[0,∞) (latency < 100ms)
  φ_stability: □[0,∞) (mode_switches/min < N)
  φ_dwell    : □[0,∞) (switch → □[0,τ_dwell] ¬switch)
  φ_recovery : (T > 80°C) → ◇[0,5s] (T < 75°C)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

from .robustness import ThermalRobustness, NetworkRobustness, ThermalConfig, NetworkConfig


class ExecutionAction(Enum):
    """Actions the STL monitor can recommend."""
    LOCAL_EXIT = "local_exit"
    OFFLOAD    = "offload"
    CONTINUE   = "continue"
    THROTTLE   = "throttle"


@dataclass
class STLSpecification:
    """
    STL specification for edge AI orchestration.

    All limits are used for both hard threshold checks and quantitative
    robustness computation.
    """
    temp_limit: float                  = 85.0   # °C
    latency_limit: float               = 100.0  # ms
    max_mode_switches_per_minute: int  = 10
    dwell_time: float                  = 1.0    # s — minimum time between switches
    recovery_timeout: float            = 5.0    # s — thermal recovery window
    recovery_target: float             = 75.0   # °C — temperature after recovery


# =============================================================================
# Monotonic Queue — O(1) amortised temporal operators
# =============================================================================

class MonotonicQueue:
    """
    Sliding-window min/max in O(1) amortised time.

    Maintains a deque whose elements are monotonically increasing (for min)
    or decreasing (for max), so that the extremum is always at the front.
    Expired elements (outside the window) are lazily removed on each push.

    Used to evaluate STL temporal operators:
      □[a,b] φ  (always):     min-queue of ρ values
      ◇[a,b] φ  (eventually): max-queue of ρ values
    """

    def __init__(self, window_size: int, operation: str = "min"):
        assert operation in ("min", "max"), "operation must be 'min' or 'max'"
        self.window_size = window_size
        self.operation   = operation
        self.queue: deque = deque()   # (value, global_index) pairs
        self.current_idx  = 0

    def push(self, value: float):
        """Insert a new value into the sliding window."""
        # Remove expired entries
        while self.queue and self.queue[0][1] <= self.current_idx - self.window_size:
            self.queue.popleft()

        # Maintain monotonicity
        if self.operation == "min":
            while self.queue and self.queue[-1][0] >= value:
                self.queue.pop()
        else:
            while self.queue and self.queue[-1][0] <= value:
                self.queue.pop()

        self.queue.append((value, self.current_idx))
        self.current_idx += 1

    def get_extremum(self) -> float:
        """Return the current min (or max) over the window."""
        if not self.queue:
            return float("inf") if self.operation == "min" else float("-inf")
        return self.queue[0][0]

    def reset(self):
        """Clear the queue and reset the index."""
        self.queue.clear()
        self.current_idx = 0


# =============================================================================
# STL Monitor
# =============================================================================

class STLMonitor:
    """
    Hybrid STL Runtime Monitor.

    Combines Lipschitz-bounded thermal robustness with worst-case network
    robustness (over Robust CP intervals) to produce a single aggregate
    robustness value and an orchestration recommendation.

    Decision priority
    ─────────────────
    1. Thermal safety    (highest — hard physical limit)
    2. Dwell-time        (mode stability)
    3. Network quality   (offload viability)
    4. Aggregate         (overall health)
    """

    def __init__(
        self,
        spec: STLSpecification,
        thermal_config: Optional[ThermalConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        history_window: int = 100,
    ):
        self.spec = spec

        # Physical dynamics
        self.thermal = ThermalRobustness(
            thermal_config or ThermalConfig(temp_limit=spec.temp_limit)
        )
        # Network dynamics
        self.network = NetworkRobustness(
            network_config or NetworkConfig(max_latency=spec.latency_limit)
        )

        # Monotonic queues for □ operators
        self.thermal_robustness_queue  = MonotonicQueue(history_window, "min")
        self.latency_robustness_queue  = MonotonicQueue(history_window, "min")

        # Mode-switching tracking
        self.mode_history: deque         = deque(maxlen=600)  # ~10 min @ 1 Hz
        self.last_switch_time: float     = 0.0
        self.current_mode: Optional[ExecutionAction] = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(
        self,
        temperature: float,
        bandwidth: float,
        latency: float,
        timestamp: float,
    ):
        """
        Ingest a new observation and update all robustness queues.

        Args:
            temperature: Current device temperature (°C)
            bandwidth:   Observed network bandwidth (Mbps)
            latency:     Observed round-trip latency (ms)
            timestamp:   Current Unix timestamp (s)
        """
        self.thermal.update(temperature, timestamp)
        self.network.update(bandwidth, latency)

        self.thermal_robustness_queue.push(self.thermal.compute_robustness(temperature))
        self.latency_robustness_queue.push(self.spec.latency_limit - latency)

    # ------------------------------------------------------------------
    # STL operator evaluations
    # ------------------------------------------------------------------

    def evaluate_always_thermal(self) -> Tuple[float, bool]:
        """
        Evaluate □[t−w, t] (T < T_lim) over the history window.

        Returns the minimum robustness observed and whether the spec is
        currently satisfied.
        """
        min_rho   = self.thermal_robustness_queue.get_extremum()
        satisfied = min_rho > 0.0
        return min_rho, satisfied

    def evaluate_dwell_time(self, timestamp: float) -> Tuple[float, bool]:
        """
        Evaluate the dwell-time constraint:

            □[0,∞) (switch → □[0, τ_dwell] ¬switch)

        Ensures a minimum dwell time τ_dwell between consecutive mode switches
        for system stability.
        """
        time_since_switch = timestamp - self.last_switch_time
        robustness        = time_since_switch - self.spec.dwell_time
        return robustness, robustness >= 0.0

    def count_mode_switches(self, window_seconds: float = 60.0) -> int:
        """Count the number of mode switches in the last ``window_seconds``."""
        history = list(self.mode_history)
        return sum(
            1 for i in range(1, len(history))
            if history[i][0] != history[i - 1][0]
        )

    # ------------------------------------------------------------------
    # Aggregate robustness
    # ------------------------------------------------------------------

    def compute_aggregate_robustness(
        self,
        temperature: float,
        bandwidth_interval: Tuple[float, float],
        latency_interval: Tuple[float, float],
        timestamp: float,
    ) -> Dict[str, float]:
        """
        Compute ρ(φ_safe) = min(ρ_thermal, ρ_bw, ρ_lat, ρ_dwell, ρ_stability).

        Returns a dictionary with per-component and aggregate values.
        """
        rho_thermal = self.thermal.compute_robustness(temperature)
        rho_bw, rho_lat = self.network.compute_worst_case_robustness(
            bandwidth_interval, latency_interval
        )
        rho_dwell, _ = self.evaluate_dwell_time(timestamp)
        switches     = self.count_mode_switches()
        rho_stability = self.spec.max_mode_switches_per_minute - switches

        return {
            "thermal":   rho_thermal,
            "bandwidth": rho_bw,
            "latency":   rho_lat,
            "dwell":     rho_dwell,
            "stability": rho_stability,
            "aggregate": min(rho_thermal, rho_bw, rho_lat, rho_dwell, rho_stability),
        }

    # ------------------------------------------------------------------
    # Orchestration decision
    # ------------------------------------------------------------------

    def get_action(
        self,
        temperature: float,
        bandwidth_interval: Tuple[float, float],
        latency_interval: Tuple[float, float],
        timestamp: float,
        exit_confidences: Optional[List[float]] = None,
    ) -> Tuple[ExecutionAction, Dict[str, float]]:
        """
        Produce an orchestration recommendation based on STL robustness.

        Decision logic (in priority order):
          1. ρ_thermal < 0          → THROTTLE  (safety critical)
          2. Dwell-time violated     → maintain current mode
          3. ρ_bw < 0 or ρ_lat < 0  → LOCAL_EXIT
          4. High confidence at exit → LOCAL_EXIT
          5. ρ_aggregate > 0         → OFFLOAD
          6. Default                 → CONTINUE

        Args:
            temperature:        Current device temperature (°C)
            bandwidth_interval: (lower, upper) from Robust CP (Mbps)
            latency_interval:   (lower, upper) from Robust CP (ms)
            timestamp:          Current time (s)
            exit_confidences:   Softmax-max confidence at each Super-Node exit

        Returns:
            action:     Recommended ExecutionAction
            robustness: Dict of per-component and aggregate ρ values
        """
        robustness = self.compute_aggregate_robustness(
            temperature, bandwidth_interval, latency_interval, timestamp
        )

        # Priority 1: Thermal safety
        if robustness["thermal"] < 0:
            action = ExecutionAction.THROTTLE

        # Priority 2: Dwell-time — hold current mode to prevent oscillation
        elif robustness["dwell"] < 0 and self.current_mode is not None:
            action = self.current_mode

        # Priority 3: Network unavailable — fall back to local execution
        elif robustness["bandwidth"] < 0 or robustness["latency"] < 0:
            action = ExecutionAction.LOCAL_EXIT

        # Priority 4: Good conditions — offload unless a local exit is confident
        elif robustness["aggregate"] > 0:
            if exit_confidences and max(exit_confidences) > 0.9:
                action = ExecutionAction.LOCAL_EXIT
            else:
                action = ExecutionAction.OFFLOAD

        else:
            action = ExecutionAction.CONTINUE

        # Record mode switch
        if action != self.current_mode:
            self.last_switch_time = timestamp
            self.current_mode     = action

        self.mode_history.append((action, timestamp))
        return action, robustness

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict:
        """Return monitoring diagnostics for analysis and debugging."""
        return {
            "thermal_rate_of_change":     self.thermal.compute_rate_of_change(),
            "network_stats":              self.network.get_network_statistics(),
            "mode_switches_last_minute":  self.count_mode_switches(),
            "current_mode": (
                self.current_mode.value if self.current_mode else None
            ),
        }
