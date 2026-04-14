"""
Super-Node: Core building block of the Mesh-Exit architecture.

Each Super-Node provides three propagation paths:
1. Local Exit Head: Immediate on-device classification with minimal latency.
2. Bottleneck Encoder: Feature compression for edge-cloud split computing.
3. Continuation Path: Standard forward propagation to the next layer (identity).

At runtime, the STL monitor selects which path to activate based on
thermal state, network conditions, and STL robustness values.
"""

import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SuperNodeConfig:
    """Configuration for a Super-Node."""
    in_channels: int
    num_classes: int
    bottleneck_dim: int = 64
    exit_hidden_dim: int = 256


class LocalExitHead(nn.Module):
    """
    Local Exit Head for immediate on-device classification.

    Minimises latency and energy consumption at the cost of potential accuracy
    loss compared to the full network. Activated when network conditions are
    poor or device temperature approaches the thermal limit.

    Architecture: AdaptiveAvgPool → Flatten → Linear → ReLU → Dropout → Linear
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        return self.classifier(self.pool(x))


class BottleneckEncoder(nn.Module):
    """
    Bottleneck Encoder for split computing.

    Compresses intermediate feature maps for low-bandwidth transmission to
    an edge server, which then completes the inference with the full network.

    Architecture: Conv1×1 (C → bottleneck_dim) → BatchNorm → ReLU
    Transmission cost: bottleneck_dim × H × W × 2 bytes  (float16)
    """

    def __init__(self, in_channels: int, bottleneck_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            compressed: Bottleneck representation [B, bottleneck_dim, H, W]
        """
        return self.encoder(x)

    def compute_transmission_size(self, x: torch.Tensor) -> int:
        """Return bytes required to transmit compressed features (float16)."""
        compressed = self.forward(x)
        return compressed.numel() * 2  # 2 bytes per float16 element


class SuperNode(nn.Module):
    """
    Super-Node: Expands the execution topology with three propagation paths.

    This is the core building block of the Mesh-Exit architecture.
    All three paths are computed on every forward pass during training
    (to allow joint optimisation). At inference, the STL monitor selects
    exactly one path per Super-Node based on real-time safety constraints.

    Outputs (4-tuple):
        exit_logits   [B, num_classes]          — local classification
        bottleneck    [B, bottleneck_dim, H, W] — compressed features for offload
        confidence    [B, 1]                    — sigmoid exit-decision score
        continuation  [B, C, H, W]             — unmodified features (identity)
    """

    def __init__(self, config: SuperNodeConfig):
        super().__init__()
        self.config = config

        # Path 1: Immediate local classification
        self.local_exit = LocalExitHead(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            hidden_dim=config.exit_hidden_dim,
        )

        # Path 2: Feature compression for split computing
        self.bottleneck = BottleneckEncoder(
            in_channels=config.in_channels,
            bottleneck_dim=config.bottleneck_dim,
        )

        # Confidence head: scalar exit-worthiness score in [0, 1]
        # Used as a fallback heuristic when the STL monitor is inactive.
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.in_channels, 1),
            nn.Sigmoid(),
        )

        # Path 3: Continuation is identity (features passed through externally)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Super-Node.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            exit_logits:  Classification logits for local exit  [B, num_classes]
            bottleneck:   Compressed features for split compute [B, bottleneck_dim, H, W]
            confidence:   Exit confidence score                 [B, 1]
            continuation: Original features (identity path)    [B, C, H, W]
        """
        exit_logits = self.local_exit(x)
        bottleneck = self.bottleneck(x)
        confidence = self.confidence_head(x)
        continuation = x  # Path 3: identity pass-through

        return exit_logits, bottleneck, confidence, continuation

    def get_exit_decision(
        self,
        x: torch.Tensor,
        threshold: float = 0.8,
    ) -> Tuple[bool, torch.Tensor]:
        """
        Heuristic exit decision based on confidence score.

        Note: In the full system this is overridden by the STL monitor.
        This is a fallback for standalone use without the safety layer.

        Args:
            x:         Input feature map
            threshold: Confidence threshold for early exit

        Returns:
            should_exit: Whether to take the local exit
            logits:      Classification logits
        """
        exit_logits, _, confidence, _ = self.forward(x)
        should_exit = confidence.mean() > threshold
        return should_exit.item(), exit_logits
