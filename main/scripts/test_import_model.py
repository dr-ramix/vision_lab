# main/scripts/test_import_forward.py
from __future__ import annotations

import torch

from fer.inference.models import (
    ResNet18,
    ResNet50,
    ConvNeXtBase,
    MobileNetV2,
    EmoCatNetsTiny,
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FER setup (your project)
    num_classes = 6
    in_channels = 3
    img_size = 64

    models = [
        ResNet18,
        ResNet50,
        ConvNeXtBase,
        MobileNetV2,
        EmoCatNetsTiny,
    ]

    for cls in models:
        print(f"\n=== Testing {cls.__name__} ===")

        # load pretrained model
        model = cls().load(device=device)
        model.eval()

        # random input: batch of 2 images
        x = torch.randn(2, in_channels, img_size, img_size, device=device)

        with torch.no_grad():
            y = model(x)

        print(f"Input shape : {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")

        # sanity checks
        assert y.shape == (2, num_classes), (
            f"{cls.__name__}: expected output shape (2,{num_classes}), got {y.shape}"
        )

        assert torch.isfinite(y).all(), (
            f"{cls.__name__}: output contains NaNs or Infs"
        )

        print("✔ Forward pass OK")

    print("\nAll models passed import + load + forward ✔")

if __name__ == "__main__":
    main()
