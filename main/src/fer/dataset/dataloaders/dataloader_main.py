from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# --------------------------------------------------
# Feste, explizite Klassen-Zuordnung (NICHT ändern)
# --------------------------------------------------
CLASS_ORDER = [
    "anger",      # 0
    "disgust",    # 1
    "fear",       # 2
    "happiness",  # 3
    "sadness",    # 4
    "surprise",   # 5
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_ORDER)}


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: Dict[str, int]


# --------------------------------------------------
# Augmentations (train only)
# p<1 => nicht jedes Bild wird augmentiert
# --------------------------------------------------
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_range=(0.08, 0.12), p: float = 0.25):
        super().__init__()
        self.sigma_range = sigma_range
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        sigma = torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]).item()
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0.0, 1.0)


def _build_transform_train():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        # Rotation ≤ 15° + Translation wenige Pixel
        transforms.RandomApply(
            [transforms.RandomAffine(
                degrees=15,
                translate=(0.02, 0.02),  # ~ wenige Pixel je nach Bildgröße
                fill=0
            )],
            p=0.35,
        ),

        # Gaussian Blur σ ≈ 1–2
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))],
            p=0.20,
        ),

        # Brightness ±0.2, Contrast ×1.5–2
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=(1.5, 2.0))],
            p=0.30,
        ),

        # Gaussian Noise σ ≈ 0.08–0.12
        AddGaussianNoise(sigma_range=(0.08, 0.12), p=0.25),
    ])


def _build_transform_eval():
    return transforms.Compose([])


# --------------------------------------------------
# NPY Dataset
# Erwartete Struktur:
#   <root>/npy/<split>/<emotion>/*.npy
# z.B. images_cropped_mtcnn/npy/test/<emotion>/abc.npy
#
# Jede npy: HxW (grau) oder HxWxC oder CxHxW.
# Ausgabe: Tensor (C,H,W) float32 in [0,1]
# --------------------------------------------------
def _as_chw_float01(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)

    # dtype -> float und ggf. skalieren
    if arr.dtype == np.uint8:
        x = torch.from_numpy(arr).float() / 255.0
    else:
        x = torch.from_numpy(arr).float()
        if x.numel() > 0 and float(x.max()) > 1.5:  # wahrscheinlich 0..255
            x = x / 255.0

    # shape normalisieren
    if x.ndim == 2:
        x = x.unsqueeze(0)  # 1xHxW
    elif x.ndim == 3:
        # HxWxC -> CxHxW
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1)
        # sonst: bereits CxHxW
    else:
        raise ValueError(f"Unsupported npy shape: {tuple(x.shape)}")

    return x.clamp(0.0, 1.0)


class NpyEmotionFolder(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        # check class folders vorhanden
        missing = [c for c in CLASS_ORDER if not (self.split_dir / c).exists()]
        if missing:
            raise FileNotFoundError(f"Missing class folders in {self.split_dir}: {missing}")

        self.samples: List[Tuple[Path, int]] = []
        for cls in CLASS_ORDER:
            cls_dir = self.split_dir / cls
            for p in sorted(cls_dir.rglob("*.npy")):
                self.samples.append((p, CLASS_TO_IDX[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found under: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        arr = np.load(path)
        x = _as_chw_float01(arr)

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def build_dataloaders(
    images_root: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoaders:
    """
    images_root muss zeigen auf:
      images_cropped_mtcnn/
        npy/train/<emotion>/*.npy
        npy/val/<emotion>/*.npy
        npy/test/<emotion>/*.npy
    """
    images_root = Path(images_root)
    npy_root = images_root / "npy"

    train_ds = NpyEmotionFolder(npy_root / "train", transform=_build_transform_train())
    val_ds   = NpyEmotionFolder(npy_root / "val",   transform=_build_transform_eval())
    test_ds  = NpyEmotionFolder(npy_root / "test",  transform=_build_transform_eval())

    pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader, class_to_idx=CLASS_TO_IDX)


# --------------------------------------------------
# Quick Sanity Check
# --------------------------------------------------
def main_quickcheck():
    project_root = Path(__file__).resolve().parents[4]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped"

    dls = build_dataloaders(images_root, batch_size=8, num_workers=0)

    xb, yb = next(iter(dls.train))
    print("class_to_idx:", dls.class_to_idx)
    print("labels in batch:", yb.tolist())
    print("input shape:", xb.shape)  # (B,C,H,W)
    print("min/max:", float(xb.min()), float(xb.max()))


if __name__ == "__main__":
    main_quickcheck()
