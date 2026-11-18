"""GPU-friendly CIFAR models used by the benchmark."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, requested: int = 8) -> int:
    """Choose a stable GroupNorm width that divides the channel count."""
    groups = min(requested, channels)
    while channels % groups:
        groups -= 1
    return groups


class GroupNormBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return torch.relu(x + residual)


class GroupNormResNet18(nn.Module):
    """CIFAR-sized ResNet-18 with GroupNorm instead of BatchNorm.

    The 3x3 stride-1 stem preserves CIFAR spatial detail. GroupNorm keeps the
    model independent of client-local batch statistics, which is important for
    federated and client-level-DP comparisons.
    """

    def __init__(self, num_classes: int = 10, base_width: int = 64) -> None:
        super().__init__()
        self.in_channels = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_width), base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 8, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [GroupNormBasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(GroupNormBasicBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x).flatten(1))


class GroupNormWideBasicBlock(nn.Module):
    """Pre-activation CIFAR WideResNet residual block with GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = torch.relu(self.norm1(x))
        residual = self.shortcut(x)
        out = self.conv1(activated)
        out = self.conv2(self.dropout(torch.relu(self.norm2(out))))
        return out + residual


class GroupNormWideResNet(nn.Module):
    """From-scratch CIFAR WideResNet with GroupNorm and no pretrained weights.

    Depth follows the CIFAR WideResNet convention ``6n + 4``. The default
    depth=28, widening=4 creates four residual blocks per stage and about
    5.85M trainable parameters for ten classes.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if depth < 10 or (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be 6n+4 and at least 10 (for example, 16 or 28).")
        if widen_factor < 1:
            raise ValueError("WideResNet widen_factor must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("WideResNet dropout must be in [0, 1).")

        blocks_per_stage = (depth - 4) // 6
        widths = (16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor)
        self.stem = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(widths[0], widths[1], blocks_per_stage, 1, dropout)
        self.stage2 = self._make_stage(widths[1], widths[2], blocks_per_stage, 2, dropout)
        self.stage3 = self._make_stage(widths[2], widths[3], blocks_per_stage, 2, dropout)
        self.final_norm = nn.GroupNorm(_group_count(widths[3]), widths[3])
        self.classifier = nn.Linear(widths[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                nn.init.zeros_(module.bias)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [GroupNormWideBasicBlock(in_channels, out_channels, stride, dropout)]
        layers.extend(
            GroupNormWideBasicBlock(out_channels, out_channels, 1, dropout)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = torch.relu(self.final_norm(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
