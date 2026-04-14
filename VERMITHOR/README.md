# Thermo-Logical Inference — Supplementary Code

This directory contains the core implementation of the methodology presented in the paper.

## Structure

```
VERMITHOR/
├── mesh_exit/                  # Mesh-Exit architecture with 3-path Super-Nodes
│   ├── super_node.py           # SuperNode: local exit, bottleneck encoder, continuation
│   └── resnet_backbone.py      # MeshExitResNet with Super-Nodes injected after layers
│
├── conformal/                  # Robust Conformal Prediction with f-divergence
│   ├── divergence_estimator.py # Neural likelihood ratio estimator + EMA tracking
│   └── conformal_predictor.py  # Standard CP and Robust CP with adaptive inflation
│
├── stl_monitor/                # STL Runtime Monitor and Integrated Controller
│   ├── robustness.py           # Lipschitz-bounded thermal + worst-case network robustness
│   ├── hybrid_dynamics.py      # Formal hybrid dynamics model + CoverageTheorem
│   ├── stl_monitor.py          # Monotonic queue, STL specifications, orchestration logic
│   ├── online_divergence.py    # Real-time O(1) f-divergence estimation + PID inflation
│   └── runtime_controller.py  # Integrated 5-mode controller tying all components together
│
└── data/
    ├── cifar10.py              # Dataset loading (CIFAR-10)
    └── cifar100.py             # Dataset loading (CIFAR-100, used in experiments)
```

## Component Overview

### Mesh-Exit Architecture (`mesh_exit/`)

The `SuperNode` is the fundamental building block, providing three propagation paths at each injection point:

1. **Local Exit Head** — Adaptive pooling + 2-layer MLP for immediate on-device classification.
2. **Bottleneck Encoder** — 1×1 convolution for feature compression prior to edge-cloud split.
3. **Continuation Path** — Identity pass-through to the next ResNet layer.

`MeshExitResNet` wraps standard ResNet-18/34/50/101 and injects Super-Nodes after configurable layer indices (default: after layers 1, 2, and 3).

### Robust Conformal Prediction (`conformal/`)

`FDivergenceEstimator` trains a binary classifier to distinguish calibration from runtime features, recovering the likelihood ratio `r(x) = p_test(x) / p_cal(x)`. This enables O(1) per-sample divergence estimation at inference time via EMA.

`RobustConformalPredictor` inflates prediction intervals according to:

```
λ(D_f) = min(λ_max,  λ_base × (1 + β × D_f))
```

guaranteeing coverage `≥ 1 − α` whenever `D_f(P_test ∥ P_cal) ≤ D_max`.

### STL Runtime Monitor (`stl_monitor/`)

**Hybrid dynamics separation**: Physical signals (temperature, battery) obey Lipschitz bounds, so their robustness is computed directly (`ρ = T_lim − T`). Network signals (latency, bandwidth) are stochastic, so robustness is evaluated worst-case over robust CP intervals.

The `MonotonicQueue` implements O(1) amortized sliding-window min/max, enabling efficient evaluation of `□[a,b]` and `◇[a,b]` STL operators.

`IntegratedRuntimeController` orchestrates five operating modes — **OFFLOAD, LOCAL_EXIT, FULL_LOCAL, THROTTLE, EMERGENCY** — under a priority ordering:

| Priority | Condition | Mode |
|---|---|---|
| 1 | `ρ_thermal < −5` | EMERGENCY |
| 2 | `ρ_thermal < 5` | THROTTLE |
| 3 | `ρ_network < 0` | LOCAL_EXIT (best-confidence exit) |
| 4 | `ρ_network < 5` | FULL_LOCAL |
| 5 | otherwise | OFFLOAD |

A dwell-time constraint prevents rapid mode oscillation; EMERGENCY and THROTTLE bypass it for immediate safety response.
