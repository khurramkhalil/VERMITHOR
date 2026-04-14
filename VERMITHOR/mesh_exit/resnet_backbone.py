"""
ResNet Backbone with Mesh-Exit Super-Node Injection.

Implements the full Mesh-Exit architecture on top of ResNet-18/34/50/101,
with Super-Nodes injected at strategic depths for adaptive inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .super_node import SuperNode, SuperNodeConfig


class ExecutionPath(Enum):
    """Execution path decisions at each Super-Node."""
    LOCAL_EXIT     = "local_exit"      # Classify on-device immediately
    SPLIT_OFFLOAD  = "split_offload"   # Send bottleneck to edge server
    CONTINUE       = "continue"        # Propagate to next ResNet layer


# =============================================================================
# ResNet Building Blocks
# =============================================================================

class BasicBlock(nn.Module):
    """Basic ResNet block — two 3×3 convolutions (used in ResNet-18/34)."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck ResNet block — 1×1 → 3×3 → 1×1 with 4× channel expansion
    (used in ResNet-50/101)."""
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


# =============================================================================
# Mesh-Exit ResNet
# =============================================================================

RESNET_CONFIGS = {
    "resnet18":  (BasicBlock,  [2, 2, 2, 2]),
    "resnet34":  (BasicBlock,  [3, 4, 6, 3]),
    "resnet50":  (Bottleneck,  [3, 4, 6, 3]),
    "resnet101": (Bottleneck,  [3, 4, 23, 3]),
}


@dataclass
class MeshExitResNetConfig:
    """Configuration for Mesh-Exit ResNet."""
    num_classes: int  = 1000
    arch: str         = "resnet50"   # resnet18 | resnet34 | resnet50 | resnet101
    bottleneck_dim: int    = 64
    exit_hidden_dim: int   = 512
    pretrained: bool       = False
    # Layers after which a Super-Node is injected (1-indexed, matching layer1..layer4).
    # Default: inject after layer1, layer2, layer3 — giving three early-exit points.
    super_node_after_layers: List[int] = field(default_factory=lambda: [1, 2, 3])


class MeshExitResNet(nn.Module):
    """
    Mesh-Exit Architecture on a ResNet backbone.

    Super-Nodes are injected after user-specified ResNet layers.  During
    a forward pass, every Super-Node computes all three paths
    (exit logits, bottleneck, confidence).  The caller may optionally supply
    ``path_decisions`` to stop at an early exit or redirect to split-offload.

    Returned dictionary keys
    ────────────────────────
    exit_logits  : list of [B, num_classes] tensors, one per Super-Node
    bottlenecks  : list of [B, bottleneck_dim, H, W] tensors
    confidences  : list of [B, 1] sigmoid scores
    features     : list of raw feature maps (detached) before each Super-Node
    logits       : [B, num_classes] — final-layer or early-exit classification
    exit_index   : int — index into exit_logits that was used (-1 = final layer)
    """

    def __init__(self, config: MeshExitResNetConfig):
        super().__init__()
        self.config = config

        if config.arch not in RESNET_CONFIGS:
            raise ValueError(f"Unknown architecture '{config.arch}'. "
                             f"Choose from {list(RESNET_CONFIGS)}")

        block, num_blocks = RESNET_CONFIGS[config.arch]
        self.expansion = block.expansion

        # ---- Stem ----
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ---- ResNet layers ----
        self._in_channels = 64
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ---- Final classifier ----
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512 * self.expansion, config.num_classes)

        # ---- Super-Nodes ----
        layer_out_channels = {
            1: 64  * self.expansion,
            2: 128 * self.expansion,
            3: 256 * self.expansion,
            4: 512 * self.expansion,
        }
        self.super_nodes = nn.ModuleDict()
        for layer_idx in config.super_node_after_layers:
            sn_cfg = SuperNodeConfig(
                in_channels=layer_out_channels[layer_idx],
                num_classes=config.num_classes,
                bottleneck_dim=config.bottleneck_dim,
                exit_hidden_dim=config.exit_hidden_dim,
            )
            self.super_nodes[f"layer{layer_idx}"] = SuperNode(sn_cfg)

        # Ordered reference for the forward loop
        self._layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_layer(self, block, mid_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self._in_channels, mid_channels, s))
            self._in_channels = mid_channels * block.expansion
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        path_decisions: Optional[List[ExecutionPath]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the Mesh-Exit ResNet.

        Args:
            x:              Input images [B, 3, H, W]
            path_decisions: Optional list of ExecutionPath decisions, one per
                            Super-Node.  When ``LOCAL_EXIT`` is requested at
                            node i, inference stops immediately and returns
                            the exit logits at that node.

        Returns:
            Dict with keys: exit_logits, bottlenecks, confidences, features,
                            logits, exit_index.
        """
        results: Dict[str, Any] = {
            "exit_logits": [],
            "bottlenecks": [],
            "confidences": [],
            "features":    [],
            "exit_index":  -1,
        }

        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        decision_idx = 0

        # Process layers, inserting Super-Node computation after each requested layer
        for layer_idx, layer in enumerate(self._layers, start=1):
            x = layer(x)

            node_key = f"layer{layer_idx}"
            if node_key in self.super_nodes:
                exit_logits, bottleneck, confidence, continuation = \
                    self.super_nodes[node_key](x)

                results["exit_logits"].append(exit_logits)
                results["bottlenecks"].append(bottleneck)
                results["confidences"].append(confidence)
                results["features"].append(x.detach())

                # Honour explicit path decision
                if path_decisions is not None and decision_idx < len(path_decisions):
                    if path_decisions[decision_idx] == ExecutionPath.LOCAL_EXIT:
                        results["logits"]     = exit_logits
                        results["exit_index"] = decision_idx
                        return results
                    # SPLIT_OFFLOAD: in deployment, the bottleneck is transmitted
                    # to the server; during training we continue locally.

                decision_idx += 1

        # Final layer classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        results["logits"] = self.fc(x)

        return results

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_exit_flops(self) -> List[float]:
        """
        Approximate cumulative GFLOPs to reach each exit point (224×224 input).

        Useful for latency-accuracy trade-off analysis.
        """
        if self.config.arch == "resnet50":
            layer_gflops = {0: 0.12, 1: 0.23, 2: 0.55, 3: 1.10, 4: 0.86}
        elif self.config.arch == "resnet18":
            layer_gflops = {0: 0.12, 1: 0.15, 2: 0.30, 3: 0.60, 4: 0.48}
        else:
            layer_gflops = {i: 0.5 for i in range(5)}

        cumulative = layer_gflops[0]
        exit_flops = []
        for layer_idx in sorted(self.config.super_node_after_layers):
            cumulative += layer_gflops[layer_idx]
            exit_flops.append(cumulative)
        exit_flops.append(sum(layer_gflops.values()))
        return exit_flops

    def get_bottleneck_sizes(self, batch_size: int = 1) -> List[int]:
        """
        Compute bottleneck transmission cost at each Super-Node in bytes (float16).

        Spatial sizes are for a 224×224 input.
        """
        spatial_sizes = {1: 56, 2: 28, 3: 14, 4: 7}
        sizes = []
        for layer_idx in self.config.super_node_after_layers:
            h = w = spatial_sizes[layer_idx]
            size = batch_size * self.config.bottleneck_dim * h * w * 2
            sizes.append(size)
        return sizes


# =============================================================================
# Factory functions
# =============================================================================

def mesh_exit_resnet18(num_classes: int = 1000, **kwargs) -> MeshExitResNet:
    """Create a Mesh-Exit ResNet-18."""
    return MeshExitResNet(MeshExitResNetConfig(num_classes=num_classes,
                                               arch="resnet18", **kwargs))


def mesh_exit_resnet50(num_classes: int = 1000, **kwargs) -> MeshExitResNet:
    """Create a Mesh-Exit ResNet-50."""
    return MeshExitResNet(MeshExitResNetConfig(num_classes=num_classes,
                                               arch="resnet50", **kwargs))


def mesh_exit_resnet101(num_classes: int = 1000, **kwargs) -> MeshExitResNet:
    """Create a Mesh-Exit ResNet-101."""
    return MeshExitResNet(MeshExitResNetConfig(num_classes=num_classes,
                                               arch="resnet101", **kwargs))
