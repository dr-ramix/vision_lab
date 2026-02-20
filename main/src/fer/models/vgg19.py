# fer/models/vgg19.py
from __future__ import annotations

import torch
import torch.nn as nn


def _conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG19(nn.Module):
    """
    VGG19-style for FER with:
      - zero downsampling in stem
      - then downsamples to reach 64 -> 32 -> 16 -> 8 (spatial)
      - VGG19 conv layout per stage: [2, 2, 4, 4] convs (total 2+2+4+4 = 12 convs)
        (Classic VGG19 has 16 convs; here we keep the 4-stage 64->32->16->8 design.)
    If you want classic VGG19 conv counts (2,2,4,4,4) we can add a 5th stage,
    but that would downsample to 4x4 if we keep pooling.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.2,
        widths: tuple[int, int, int, int] = (64, 128, 256, 512),
        layers: tuple[int, int, int, int] = (2, 2, 4, 4),
    ) -> None:
        super().__init__()
        assert len(widths) == 4
        assert len(layers) == 4

        w0, w1, w2, w3 = widths
        l0, l1, l2, l3 = layers

        # Stem (NO downsampling): convs at full res (e.g., 64x64)
        stem: list[nn.Module] = []
        ch = in_channels
        for _ in range(l0):
            stem.append(_conv3x3(ch, w0))
            ch = w0
        self.stem = nn.Sequential(*stem)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64 -> 32
        self.stage1 = self._make_stage(ch, w1, l1)
        ch = w1

        # 32 -> 16
        self.stage2 = self._make_stage(ch, w2, l2)
        ch = w2

        # 16 -> 8
        self.stage3 = self._make_stage(ch, w3, l3)
        ch = w3

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(ch, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_layers: int) -> nn.Sequential:
        blocks: list[nn.Module] = []
        ch = in_ch
        for _ in range(n_layers):
            blocks.append(_conv3x3(ch, out_ch))
            ch = out_ch
        return nn.Sequential(*blocks)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # no downsample
        x = self.pool(x)  # 64 -> 32
        x = self.stage1(x)
        x = self.pool(x)  # 32 -> 16
        x = self.stage2(x)
        x = self.pool(x)  # 16 -> 8
        x = self.stage3(x)
        return self.head(x)
