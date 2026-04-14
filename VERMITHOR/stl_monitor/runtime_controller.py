"""
Integrated Runtime Controller for Thermo-Logical Inference.

This module is the central integration point that ties all components together:

  1. Mesh-Exit ResNet      — adaptive inference with three-path Super-Nodes
  2. STL Monitor           — formal verification of thermal and network specs
  3. Robust CP             — coverage-guaranteed prediction intervals
  4. Online f-Divergence   — real-time distribution shift detection

The controller maps the current system state to one of five operating modes:

  ┌───────────────┬──────────────────────────────────────────────────────┐
  │ Mode          │ Description                                          │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ OFFLOAD       │ Transmit bottleneck to server; best accuracy         │
  │ LOCAL_EXIT    │ Classify at best-confidence Super-Node exit          │
  │ FULL_LOCAL    │ Run complete ResNet on-device                        │
  │ THROTTLE      │ Reduce workload; approaching thermal limit           │
  │ EMERGENCY     │ Stop heavy computation; thermal limit exceeded       │
  └───────────────┴──────────────────────────────────────────────────────┘

Decision priority
──────────────────
  ρ_thermal < −5  →  EMERGENCY  (immediate safety response)
  ρ_thermal < 5   →  THROTTLE   (proactive thermal management)
  ρ_network < 0   →  LOCAL_EXIT (network too poor to offload)
  ρ_network < 5   →  FULL_LOCAL (marginal network conditions)
  otherwise       →  OFFLOAD    (all constraints satisfied)

A dwell-time constraint prevents rapid oscillation between modes;
EMERGENCY and THROTTLE bypass it for safety.
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .hybrid_dynamics import (
    HybridSystemSpec,
    HybridRobustnessComputer,
    CoverageTheorem,
    create_default_hybrid_spec,
)
from .online_divergence import (
    OnlineFDivergenceEstimator,
    OnlineDivergenceConfig,
    AdaptiveInflationController,
)
from .stl_monitor import STLMonitor, STLSpecification


# =============================================================================
# Data classes
# =============================================================================

class SystemMode(Enum):
    """Operating mode of the integrated runtime system."""
    OFFLOAD    = "offload"      # Split computing: bottleneck → server
    LOCAL_EXIT = "local_exit"   # Early exit on-device at best Super-Node
    FULL_LOCAL = "full_local"   # Complete ResNet inference on-device
    THROTTLE   = "throttle"     # Reduce workload (thermal approaching limit)
    EMERGENCY  = "emergency"    # Stop heavy computation (limit exceeded)


@dataclass
class RuntimeState:
    """Snapshot of system state at one control step."""
    # Physical sensors
    temperature: float   # °C
    battery: float       # %

    # Network estimates from Robust CP
    latency_mean: float                       # ms
    latency_interval: Tuple[float, float]     # (lower, upper) ms
    bandwidth_mean: float                     # Mbps
    bandwidth_interval: Tuple[float, float]   # (lower, upper) Mbps

    # Distribution shift
    divergence: float    # D_f(P_test ∥ P_cal)

    # Timing
    timestamp: float     # Unix timestamp

    # Model state
    exit_confidences: Optional[List[float]] = None  # Softmax-max at each exit


@dataclass
class RuntimeDecision:
    """Decision produced by the integrated controller at one control step."""
    mode: SystemMode
    exit_index: int                      # Index into exit_logits to use; −1 for final
    robustness: float                    # Overall hybrid STL robustness
    robustness_breakdown: Dict[str, float]
    inflation_factor: float              # Current λ(D_f)
    safe_horizon_s: float                # Guaranteed safe time window (s)
    reasoning: str                       # Human-readable explanation


# =============================================================================
# Integrated Runtime Controller
# =============================================================================

class IntegratedRuntimeController:
    """
    Main controller integrating all Thermo-Logical Inference components.

    At each control step the controller:
      1. Estimates f-divergence from runtime network features.
      2. Computes inflated CP intervals for latency and bandwidth.
      3. Evaluates hybrid STL robustness (physical + network).
      4. Selects an operating mode via the priority table.
      5. Applies dwell-time constraints to prevent rapid mode switching.
      6. Returns a RuntimeDecision with formal guarantees.

    Key theoretical property: the separation of physical (Lipschitz) and
    network (stochastic) dynamics makes real-time formal verification tractable
    even on resource-constrained edge hardware.
    """

    def __init__(
        self,
        network: nn.Module,
        spec: Optional[HybridSystemSpec] = None,
        feature_dim: int = 32,
    ):
        """
        Args:
            network:     Mesh-Exit ResNet for inference
            spec:        Hybrid system specification (uses defaults if None)
            feature_dim: Dimension of network features for divergence estimation
        """
        self.network = network
        self.spec    = spec or create_default_hybrid_spec()

        # --- Core sub-systems ---
        self.robustness_computer = HybridRobustnessComputer(self.spec)
        self.coverage_theorem    = CoverageTheorem(alpha=0.1, beta=0.5)

        # Online divergence estimation
        self.divergence_estimator = OnlineFDivergenceEstimator(
            input_dim=feature_dim,
            config=OnlineDivergenceConfig(ema_alpha=0.1, buffer_size=500),
        )

        # Adaptive interval inflation
        self.inflation_controller = AdaptiveInflationController(
            lambda_max=self.spec.latency.inflation_max if self.spec.latency else 3.0,
            beta=self.spec.latency.inflation_base      if self.spec.latency else 0.5,
        )

        # STL monitor (uses robustness and dwell constraints)
        self.stl_monitor = STLMonitor(STLSpecification(
            temp_limit=self.spec.temp_limit,
            latency_limit=self.spec.max_latency,
        ))

        # --- State tracking ---
        self.current_state: Optional[RuntimeState] = None
        self.current_mode:  SystemMode             = SystemMode.OFFLOAD
        self.mode_dwell_count: int                 = 0
        self.history: List[RuntimeDecision]        = []

        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "decisions_made":     0,
            "mode_switches":      0,
            "thermal_violations": 0,
            "network_violations": 0,
            "coverage_maintained": True,
        }

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(
        self,
        temperature: float,
        battery: float,
        network_features: torch.Tensor,
        exit_confidences: Optional[List[float]] = None,
    ) -> RuntimeState:
        """
        Ingest sensor readings and produce an updated RuntimeState.

        Estimates the current f-divergence, computes inflated CP intervals
        for latency and bandwidth, and packages everything into RuntimeState.

        Args:
            temperature:       Current device temperature (°C)
            battery:           Current battery level (%)
            network_features:  Features for divergence estimation [D] or [1, D]
            exit_confidences:  Softmax-max confidence at each Super-Node exit

        Returns:
            Updated RuntimeState
        """
        if network_features.dim() == 1:
            network_features = network_features.unsqueeze(0)

        # --- Divergence estimation ---
        divergence = self.divergence_estimator.estimate_divergence(
            network_features, update_ema=True
        ).item()

        # --- Inflated CP intervals ---
        lat_base = (self.spec.latency.calibration_quantiles
                    if self.spec.latency else (10.0, 50.0))
        bw_base  = (self.spec.bandwidth.calibration_quantiles
                    if self.spec.bandwidth else (20.0, 100.0))

        lat_interval = self.inflation_controller.get_inflated_interval(lat_base, divergence)
        bw_interval  = self.inflation_controller.get_inflated_interval(bw_base,  divergence)

        state = RuntimeState(
            temperature=temperature,
            battery=battery,
            latency_mean=(lat_interval[0] + lat_interval[1]) / 2.0,
            latency_interval=lat_interval,
            bandwidth_mean=(bw_interval[0] + bw_interval[1]) / 2.0,
            bandwidth_interval=bw_interval,
            divergence=divergence,
            timestamp=time.time(),
            exit_confidences=exit_confidences,
        )

        self.current_state = state
        return state

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def make_decision(
        self,
        state: Optional[RuntimeState] = None,
    ) -> RuntimeDecision:
        """
        Produce a scheduling decision from the current RuntimeState.

        Steps:
          1. Compute hybrid robustness (physical + network).
          2. Select mode via priority ordering.
          3. Apply dwell-time constraint.
          4. Update metrics and history.
          5. Return RuntimeDecision.

        Args:
            state: RuntimeState to use (defaults to self.current_state)

        Returns:
            RuntimeDecision with mode, exit index, robustness, and reasoning
        """
        state = state or self.current_state
        if state is None:
            raise ValueError("No state available. Call update_state() first.")

        # Hybrid robustness
        hybrid_rob, breakdown = self.robustness_computer.compute_hybrid_robustness(
            temperature=state.temperature,
            latency_interval=state.latency_interval,
            bandwidth_interval=state.bandwidth_interval,
            battery=state.battery,
        )
        safe_horizon = self.robustness_computer.compute_physical_horizon(
            state.temperature, state.battery
        )

        # Mode selection
        mode, exit_idx, reasoning = self._select_mode(state, breakdown, safe_horizon)

        # Dwell-time constraint
        mode, reasoning = self._apply_dwell_constraint(mode, reasoning)

        # Metrics
        self.metrics["decisions_made"] += 1
        if mode != self.current_mode:
            self.metrics["mode_switches"] += 1
            self.mode_dwell_count  = 0
            self.current_mode      = mode
        else:
            self.mode_dwell_count += 1

        if breakdown["thermal"] < 0:
            self.metrics["thermal_violations"] += 1
        if breakdown["network"] < 0:
            self.metrics["network_violations"] += 1

        inflation = self.inflation_controller.compute_inflation(state.divergence)

        decision = RuntimeDecision(
            mode=mode,
            exit_index=exit_idx,
            robustness=hybrid_rob,
            robustness_breakdown=breakdown,
            inflation_factor=inflation,
            safe_horizon_s=safe_horizon,
            reasoning=reasoning,
        )

        self.history.append(decision)
        return decision

    # ------------------------------------------------------------------
    # Mode selection helpers
    # ------------------------------------------------------------------

    def _select_mode(
        self,
        state: RuntimeState,
        breakdown: Dict[str, float],
        safe_horizon: float,
    ) -> Tuple[SystemMode, int, str]:
        """Apply the priority table to select an operating mode."""

        # P1: EMERGENCY — thermal limit exceeded by ≥ 5°C
        if breakdown["thermal"] < -5.0:
            return (SystemMode.EMERGENCY, 0,
                    "EMERGENCY: Thermal limit exceeded by 5°C")

        # P2: THROTTLE — within 5°C of limit
        if breakdown["thermal"] < 5.0:
            return (SystemMode.THROTTLE, 0,
                    f"Throttling: only {breakdown['thermal']:.1f}°C margin remaining")

        # P3: LOCAL_EXIT — network too poor to offload
        if breakdown["network"] < 0:
            best_exit = (
                max(range(len(state.exit_confidences)),
                    key=lambda i: state.exit_confidences[i])
                if state.exit_confidences else 0
            )
            return (SystemMode.LOCAL_EXIT, best_exit,
                    f"Local exit: network robustness {breakdown['network']:.1f} < 0")

        # P4: FULL_LOCAL — marginal network
        if breakdown["network"] < 5.0:
            return (SystemMode.FULL_LOCAL, -1,
                    f"Full local: network robustness only {breakdown['network']:.1f}")

        # P5: OFFLOAD — all constraints satisfied
        return (SystemMode.OFFLOAD, -1,
                f"Offload: hybrid robustness {breakdown['hybrid']:.1f}")

    def _apply_dwell_constraint(
        self,
        proposed_mode: SystemMode,
        reasoning: str,
    ) -> Tuple[SystemMode, str]:
        """
        Enforce the STL dwell-time constraint to prevent mode oscillation.

        STL spec: □[0,∞) (switch → □[0, τ_min] ¬switch)

        EMERGENCY and THROTTLE always bypass dwell constraints for safety.
        """
        min_dwell = 5  # Minimum control steps before switching

        if proposed_mode in (SystemMode.EMERGENCY, SystemMode.THROTTLE):
            return proposed_mode, reasoning

        if (self.mode_dwell_count < min_dwell
                and proposed_mode != self.current_mode):
            return (
                self.current_mode,
                f"Dwell constraint: staying in {self.current_mode.value} "
                f"({self.mode_dwell_count}/{min_dwell} steps)",
            )

        return proposed_mode, reasoning

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def add_calibration_data(
        self,
        latency_samples: torch.Tensor,
        bandwidth_samples: torch.Tensor,
    ):
        """
        Provide calibration measurements for the online divergence estimator.

        Args:
            latency_samples:   Latency observations from calibration period [N]
            bandwidth_samples: Bandwidth observations from calibration period [N]
        """
        features = torch.stack([latency_samples, bandwidth_samples], dim=-1)
        self.divergence_estimator.add_calibration_data(features)

    # ------------------------------------------------------------------
    # End-to-end inference
    # ------------------------------------------------------------------

    def run_inference(
        self,
        images: torch.Tensor,
        temperature: float,
        battery: float,
        latency_obs: float,
        bandwidth_obs: float,
    ) -> Dict[str, Any]:
        """
        Execute the complete inference pipeline with safety monitoring.

        Performs a forward pass through the Mesh-Exit network, updates the
        runtime state, makes a scheduling decision, and selects the
        appropriate output (early exit or final layer) based on the decision.

        Args:
            images:        Input images [B, 3, H, W]
            temperature:   Current device temperature (°C)
            battery:       Battery level (%)
            latency_obs:   Observed network round-trip time (ms)
            bandwidth_obs: Observed network bandwidth (Mbps)

        Returns:
            Dict with keys: predictions, decision, state, exit_confidences,
                            network_output
        """
        network_features = torch.tensor(
            [latency_obs, bandwidth_obs], dtype=torch.float32
        )

        with torch.no_grad():
            output = self.network(images)

        # Softmax-max confidence at each Super-Node exit
        confidences = [
            torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
            for logits in output["exit_logits"]
        ]

        # Update state and make decision
        state    = self.update_state(temperature, battery, network_features, confidences)
        decision = self.make_decision(state)

        # Select prediction based on decision
        if 0 <= decision.exit_index < len(output["exit_logits"]):
            predictions = output["exit_logits"][decision.exit_index]
        else:
            predictions = output["logits"]

        return {
            "predictions":     predictions,
            "decision":        decision,
            "state":           state,
            "exit_confidences": confidences,
            "network_output":  output,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information for analysis and debugging."""
        div_mean, div_std = self.divergence_estimator.get_divergence_with_uncertainty()
        return {
            "metrics":         self.metrics.copy(),
            "current_mode":    self.current_mode.value,
            "mode_dwell_count": self.mode_dwell_count,
            "divergence_mean": div_mean,
            "divergence_std":  div_std,
            "inflation_factor": self.inflation_controller.prev_lambda,
            "history_length":  len(self.history),
            "is_calibrated":   self.divergence_estimator.is_calibrated,
        }

    def reset(self):
        """Reset all controller state."""
        self.current_state    = None
        self.current_mode     = SystemMode.OFFLOAD
        self.mode_dwell_count = 0
        self.history.clear()
        self.metrics = {
            "decisions_made":      0,
            "mode_switches":       0,
            "thermal_violations":  0,
            "network_violations":  0,
            "coverage_maintained": True,
        }
        self.divergence_estimator.reset()
        self.inflation_controller.reset()
