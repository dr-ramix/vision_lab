import torch.nn as nn
import os
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ResBlock(nn.Module):
    """
    Basic Residual Block (ResNet-18 / ResNet-34 style)

    Main path:
      Conv3x3(stride) -> BN -> ReLU -> Conv3x3(stride=1) -> BN
    Skip path:
      Identity OR (1x1 conv with stride) -> BN
    Output:
      ReLU(main + skip)
    """
    expansion = 1

    def __init__(self, input_channel, output_channel, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(output_channel)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3,                  stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(output_channel)

        self.downsample = None
        if stride != 1 or input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F




class ResNet18FER(nn.Module):
    """
    ResNet-18 for FER (RGB, 64x64) using the provided ResBlock.

    Stages:
      stem: 3x3 conv, 64 channels
      layer1: 2 blocks, 64 ch, stride 1
      layer2: 2 blocks, 128 ch, first block stride 2
      layer3: 2 blocks, 256 ch, first block stride 2
      layer4: 2 blocks, 512 ch, first block stride 2
      head: global average pool + FC -> num_classes
    """
    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  blocks=2, stride=1)  # 64x64
        self.layer2 = self._make_layer(128, blocks=2, stride=2)  # 32x32
        self.layer3 = self._make_layer(256, blocks=2, stride=2)  # 16x16
        self.layer4 = self._make_layer(512, blocks=2, stride=2)  # 8x8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # -> (N, 512, 1, 1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(self.inplanes, out_channels, stride=stride))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.inplanes, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # x: (N, 3, 64, 64)
        x = self.conv1(x)   # (N, 64, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # (N, 64, 64, 64)
        x = self.layer2(x)  # (N, 128, 32, 32)
        x = self.layer3(x)  # (N, 256, 16, 16)
        x = self.layer4(x)  # (N, 512, 8, 8)

        x = self.avgpool(x)         # (N, 512, 1, 1)
        x = torch.flatten(x, 1)     # (N, 512)
        x = self.fc(x)              # (N, 7)
        return x

