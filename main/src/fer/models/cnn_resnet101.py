import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block (ResNet-50 / ResNet-101 / ResNet-152 style)

    Main path:
      Conv1x1(reduce) -> BN -> GELU ->
      Conv3x3(stride) -> BN -> GELU ->
      Conv1x1(expand) -> BN

    Skip path:
      Identity OR (1x1 conv with stride) -> BN

    Output:
      GELU(main + skip)

    Notes:
      - expansion=4 (output channels = base_channels * 4)
    """
    expansion = 4

    def __init__(self, input_channel: int, base_channel: int, stride: int = 1):
        super().__init__()
        out_channel = base_channel * self.expansion

        self.conv1 = nn.Conv2d(input_channel, base_channel, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(base_channel)
        self.act   = nn.GELU()

        self.conv2 = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(base_channel)

        self.conv3 = nn.Conv2d(base_channel, out_channel, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channel)

        self.downsample = None
        if stride != 1 or input_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNet101FER(nn.Module):
    """
    ResNet-101 (adapted for 64x64 input images, GELU activations)

    Stem:
      Conv3x3(stride=1) -> BN -> GELU
      (No initial maxpool to avoid shrinking too early for 64x64)

    Stages (ResNet-101 layout):
      conv2_x: 3  bottleneck blocks, base=64,  stride=1
      conv3_x: 4  bottleneck blocks, base=128, stride=2
      conv4_x: 23 bottleneck blocks, base=256, stride=2
      conv5_x: 3  bottleneck blocks, base=512, stride=2

    Head:
      Global AvgPool -> FC(num_classes)

    Input:  (B, in_channels, 64, 64)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64

        # Stem (64 -> 64)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.act   = nn.GELU()

        # ResNet-101 stages: [3, 4, 23, 3]
        self.layer1 = self._make_layer(base_channel=64,  blocks=3,  stride=1)  # 64x64
        self.layer2 = self._make_layer(base_channel=128, blocks=4,  stride=2)  # 32x32
        self.layer3 = self._make_layer(base_channel=256, blocks=23, stride=2)  # 16x16
        self.layer4 = self._make_layer(base_channel=512, blocks=3,  stride=2)  # 8x8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def _make_layer(self, base_channel: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []

        layers.append(BottleneckBlock(self.inplanes, base_channel, stride=stride))
        self.inplanes = base_channel * BottleneckBlock.expansion

        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.inplanes, base_channel, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Quick sanity test
if __name__ == "__main__":
    model = ResNet101FER(num_classes=6, in_channels=3)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print("Output shape:", y.shape)
