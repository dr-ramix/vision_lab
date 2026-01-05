
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Aktivierungsfunktionen
# ----------------------------

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


# ----------------------------
# Squeeze-and-Excitation Block
# ----------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.act = HSigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = self.act(self.fc2(scale))
        return x * scale


# ----------------------------
# Inverted Residual Block (V3)
# ----------------------------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand, use_se, use_hs):
        super().__init__()
        hidden_dim = int(inp * expand)
        self.use_residual = stride == 1 and inp == oup

        activation = HSwish() if use_hs else nn.ReLU(inplace=True)

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
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride,
                      kernel // 2, groups=hidden_dim, bias=False),
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

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# ----------------------------
# MobileNetV3 Large
# ----------------------------

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        # Feature blocks (official V3 Large config)
        self.features = nn.Sequential(
            InvertedResidual(16,  16, 3, 1, 1,   False, False),
            InvertedResidual(16,  24, 3, 2, 4,   False, False),
            InvertedResidual(24,  24, 3, 1, 3,   False, False),
            InvertedResidual(24,  40, 5, 2, 3,   True,  False),
            InvertedResidual(40,  40, 5, 1, 3,   True,  False),
            InvertedResidual(40,  40, 5, 1, 3,   True,  False),
            InvertedResidual(40,  80, 3, 2, 6,   False, True),
            InvertedResidual(80,  80, 3, 1, 2.5, False, True),
            InvertedResidual(80,  80, 3, 1, 2.3, False, True),
            InvertedResidual(80, 112, 3, 1, 6,   True,  True),
            InvertedResidual(112,112, 3, 1, 6,   True,  True),
            InvertedResidual(112,160, 5, 2, 6,   True,  True),
            InvertedResidual(160,160, 5, 1, 6,   True,  True),
            InvertedResidual(160,160, 5, 1, 6,   True,  True),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            HSwish()
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
