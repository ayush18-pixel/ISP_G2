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


class GroupNormBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        bottleneck_channels = out_channels
        expanded_channels = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(bottleneck_channels), bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(bottleneck_channels), bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, expanded_channels, 1, bias=False)
        self.norm3 = nn.GroupNorm(_group_count(expanded_channels), expanded_channels)
        if stride != 1 or in_channels != expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(expanded_channels), expanded_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = torch.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return torch.relu(x + residual)


class GroupNormResNet(nn.Module):
    """CIFAR-sized ResNet with GroupNorm instead of BatchNorm.

    The 3x3 stride-1 stem preserves CIFAR spatial detail. GroupNorm keeps the
    model independent of client-local batch statistics, which is important for
    federated and client-level-DP comparisons.
    """

    def __init__(
        self,
        block: type[GroupNormBasicBlock] | type[GroupNormBottleneckBlock],
        layers: tuple[int, int, int, int],
        num_classes: int = 10,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_width), base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 8 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[GroupNormBasicBlock] | type[GroupNormBottleneckBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        layers.extend(block(self.in_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x).flatten(1))


class GroupNormResNet18(GroupNormResNet):
    def __init__(self, num_classes: int = 10, base_width: int = 64) -> None:
        super().__init__(GroupNormBasicBlock, (2, 2, 2, 2), num_classes, base_width)


class GroupNormResNet50(GroupNormResNet):
    def __init__(self, num_classes: int = 10, base_width: int = 64) -> None:
        super().__init__(GroupNormBottleneckBlock, (3, 4, 6, 3), num_classes, base_width)
