# main/src/fer/models/mobilenetv3.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6


class HSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3, inplace=True) / 6


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.act = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = self.act(self.fc2(scale))
        return x * scale


class InvertedResidualV3(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int, stride: int, expand: float, use_se: bool, use_hs: bool):
        super().__init__()
        hidden_dim = int(round(inp * expand))
        self.use_residual = (stride == 1 and inp == oup)

        activation: nn.Module = HSwish() if use_hs else nn.ReLU(inplace=True)

        layers = []

        # Expansion
        if expand != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation,
            ]

        # Depthwise
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation,
        ]

        # SE
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # Projection
        layers += [
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3LargeScratch(nn.Module):
    """
    MobileNetV3 Large from scratch
    """
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish(),
        )

        # Feature blocks (official-ish V3 Large config)
        self.features = nn.Sequential(
            InvertedResidualV3(16,  16, 3, 1, 1.0, False, False),
            InvertedResidualV3(16,  24, 3, 2, 4.0, False, False),
            InvertedResidualV3(24,  24, 3, 1, 3.0, False, False),
            InvertedResidualV3(24,  40, 5, 2, 3.0, True,  False),
            InvertedResidualV3(40,  40, 5, 1, 3.0, True,  False),
            InvertedResidualV3(40,  40, 5, 1, 3.0, True,  False),
            InvertedResidualV3(40,  80, 3, 2, 6.0, False, True),
            InvertedResidualV3(80,  80, 3, 1, 2.5, False, True),
            InvertedResidualV3(80,  80, 3, 1, 2.3, False, True),
            InvertedResidualV3(80, 112, 3, 1, 6.0, True,  True),
            InvertedResidualV3(112,112, 3, 1, 6.0, True,  True),
            InvertedResidualV3(112,160, 5, 2, 6.0, True,  True),
            InvertedResidualV3(160,160, 5, 1, 6.0, True,  True),
            InvertedResidualV3(160,160, 5, 1, 6.0, True,  True),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            HSwish(),
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def mobilenetv3_large_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    # transfer ignored (scratch model) but kept for registry signature compatibility
    return MobileNetV3LargeScratch(num_classes=num_classes, in_channels=in_channels)
