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
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.act = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = self.act(self.fc2(scale))
        return x * scale


class InvertedResidualV3(nn.Module):
    """
    MobileNetV3-style inverted residual block.

    Downsampling (if any) happens only in the depthwise conv via `stride`.
    Residual connection is used only when stride=1 and inp==oup.
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel: int,
        stride: int,
        expand: float,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(inp * expand))
        self.use_residual = (stride == 1 and inp == oup)

        activation: nn.Module = HSwish() if use_hs else nn.ReLU(inplace=True)

        layers: list[nn.Module] = []

        # Expansion (1x1)
        if expand != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation,
            ]
        else:
            hidden_dim = inp

        # Depthwise (k x k), stride controls downsampling
        layers += [
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            activation,
        ]

        # Squeeze-and-Excitation (optional)
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # Projection (1x1)
        layers += [
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


# ----------------------------
# Channel scaling helper
# ----------------------------
def _make_divisible(v: float, divisor: int = 8) -> int:
    v = int(v + divisor / 2) // divisor * divisor
    return max(divisor, v)


class _MobileNetV3LargeBase(nn.Module):
    """
    MobileNetV3-ish backbone for FER.
    Modified downsampling schedule (for 64x64 inputs):

      No downsampling in stem: 64 -> 64
      Three stride-2 blocks in features: 64 -> 32 -> 16 -> 8
      No further downsampling afterwards.

    Head uses global average pooling to 1x1.
    """
    def __init__(self, num_classes: int = 6, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        def c(ch: int) -> int:
            return _make_divisible(ch * width_mult, 8)

        # Stem (NO downsampling): stride=1 keeps 64x64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(16), 3, stride=1, padding=1, bias=False),  # changed stride=2 -> 1
            nn.BatchNorm2d(c(16)),
            HSwish(),
        )

        # Feature blocks (downsampling only at 3 blocks => 64 -> 32 -> 16 -> 8)
        self.features = nn.Sequential(
            # 64x64
            InvertedResidualV3(c(16),  c(16), 3, 1, 1.0, False, False),

            # 64 -> 32
            InvertedResidualV3(c(16),  c(24), 3, 2, 4.0, False, False),
            # 32
            InvertedResidualV3(c(24),  c(24), 3, 1, 3.0, False, False),

            # 32 -> 16
            InvertedResidualV3(c(24),  c(40), 5, 2, 3.0, True,  False),
            # 16
            InvertedResidualV3(c(40),  c(40), 5, 1, 3.0, True,  False),
            InvertedResidualV3(c(40),  c(40), 5, 1, 3.0, True,  False),

            # 16 -> 8
            InvertedResidualV3(c(40),  c(80), 3, 2, 6.0, False, True),
            # 8
            InvertedResidualV3(c(80),  c(80), 3, 1, 2.5, False, True),
            InvertedResidualV3(c(80),  c(80), 3, 1, 2.3, False, True),

            # 8
            InvertedResidualV3(c(80),  c(112), 3, 1, 6.0, True,  True),
            InvertedResidualV3(c(112), c(112), 3, 1, 6.0, True,  True),

            # Was 8 -> 4 with stride=2, now keep 8 by using stride=1
            InvertedResidualV3(c(112), c(160), 5, 1, 6.0, True,  True),  # changed stride=2 -> 1
            # 8
            InvertedResidualV3(c(160), c(160), 5, 1, 6.0, True,  True),
            InvertedResidualV3(c(160), c(160), 5, 1, 6.0, True,  True),
        )

        # Head (kept as-is, global pool from 8x8 -> 1x1)
        head_in = c(160)
        head_mid = c(960)

        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_mid, 1, bias=False),
            nn.BatchNorm2d(head_mid),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(head_mid, 1280, 1),
            HSwish(),
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)       # 64x64 -> 64x64
        x = self.features(x)   # 64 -> 32 -> 16 -> 8
        x = self.head(x)       # 8x8 -> 1x1
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# Sizes (explicit classes)
# ============================================================

class MobileNetV3TinyScratch(_MobileNetV3LargeBase):
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__(num_classes=num_classes, in_channels=in_channels, width_mult=0.35)


class MobileNetV3SmallScratch(_MobileNetV3LargeBase):
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__(num_classes=num_classes, in_channels=in_channels, width_mult=0.50)


class MobileNetV3BaseScratch(_MobileNetV3LargeBase):
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__(num_classes=num_classes, in_channels=in_channels, width_mult=0.75)


class MobileNetV3LargeScratch(_MobileNetV3LargeBase):
    """
    “Large” (width_mult=1.0)
    """
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__(num_classes=num_classes, in_channels=in_channels, width_mult=1.0)


class MobileNetV3XLargeScratch(_MobileNetV3LargeBase):
    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__(num_classes=num_classes, in_channels=in_channels, width_mult=1.25)


# ============================================================
# Factories (registry-compatible)
# ============================================================

def mobilenetv3_tiny_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    return MobileNetV3TinyScratch(num_classes=num_classes, in_channels=in_channels)


def mobilenetv3_small_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    return MobileNetV3SmallScratch(num_classes=num_classes, in_channels=in_channels)


def mobilenetv3_base_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    return MobileNetV3BaseScratch(num_classes=num_classes, in_channels=in_channels)


def mobilenetv3_large_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    return MobileNetV3LargeScratch(num_classes=num_classes, in_channels=in_channels)


def mobilenetv3_xlarge_fer(*, num_classes: int, in_channels: int = 3, transfer: bool = False, **_) -> nn.Module:
    return MobileNetV3XLargeScratch(num_classes=num_classes, in_channels=in_channels)


# ============================================================
# quick shape test
# ============================================================
if __name__ == "__main__":
    m = MobileNetV3LargeScratch(num_classes=6, in_channels=3)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    print("out:", y.shape)  # [2, 6]
