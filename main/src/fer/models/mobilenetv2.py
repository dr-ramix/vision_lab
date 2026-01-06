# main/src/fer/models/mobilenetv2.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class InvertedResidualV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1x1 Expansion
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        # 3x3 Depthwise
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )

        # 1x1 Projection
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Scratch(nn.Module):
    """
    MobileNetV2 from scratch
    Input: 64x64 (or any) RGB by default
    Output: num_classes
    """
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # 64x64 -> 32x32
            conv_bn_relu(in_channels, 32, kernel_size=3, stride=2, padding=1),

            InvertedResidualV2(32, 16, stride=1, expand_ratio=1),

            # 32x32 -> 16x16
            InvertedResidualV2(16, 24, stride=2, expand_ratio=6),
            InvertedResidualV2(24, 24, stride=1, expand_ratio=6),

            # 16x16 -> 8x8
            InvertedResidualV2(24, 32, stride=2, expand_ratio=6),
            InvertedResidualV2(32, 32, stride=1, expand_ratio=6),
            InvertedResidualV2(32, 32, stride=1, expand_ratio=6),

            # 8x8 -> 4x4
            InvertedResidualV2(32, 64, stride=2, expand_ratio=6),
            InvertedResidualV2(64, 64, stride=1, expand_ratio=6),
            InvertedResidualV2(64, 64, stride=1, expand_ratio=6),
            InvertedResidualV2(64, 64, stride=1, expand_ratio=6),

            InvertedResidualV2(64, 96, stride=1, expand_ratio=6),
            InvertedResidualV2(96, 96, stride=1, expand_ratio=6),
            InvertedResidualV2(96, 96, stride=1, expand_ratio=6),

            # 4x4 -> 2x2
            InvertedResidualV2(96, 160, stride=2, expand_ratio=6),
            InvertedResidualV2(160, 160, stride=1, expand_ratio=6),
            InvertedResidualV2(160, 160, stride=1, expand_ratio=6),

            InvertedResidualV2(160, 320, stride=1, expand_ratio=6),
        )

        self.conv_last = conv_bn_relu(320, 1280, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenetv2_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    # transfer ignored (scratch model) but kept for registry signature compatibility
    return MobileNetV2Scratch(num_classes=num_classes, in_channels=in_channels)
