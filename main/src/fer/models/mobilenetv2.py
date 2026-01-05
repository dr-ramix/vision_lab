# ----------------------------
# MobileNetV2 from Scratch
# Input: 64x64 RGB
# Output: 6 Klassen
# ----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Hilfsfunktion: Conv + BN + ReLU6
# ----------------------------
def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# ----------------------------
# Inverted Residual Block
# ----------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1x1 Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # 3x3 Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # 1x1 Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ----------------------------
# MobileNetV2 Architektur
# ----------------------------
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            # 64x64 -> 32x32
            conv_bn_relu(3, 32, kernel_size=3, stride=2, padding=1),

            InvertedResidual(32, 16, stride=1, expand_ratio=1),

            # 32x32 -> 16x16
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),

            # 16x16 -> 8x8
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),

            # 8x8 -> 4x4
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),

            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),

            # 4x4 -> 2x2
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),

            InvertedResidual(160, 320, stride=1, expand_ratio=6),
        )

        self.conv_last = conv_bn_relu(320, 1280, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ----------------------------
# Testlauf (optional)
# ----------------------------
if __name__ == "__main__":
    model = MobileNetV2(num_classes=6)
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Erwartet: [1, 6]
