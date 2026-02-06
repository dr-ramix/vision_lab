from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        # clamp(0,1) weil wir vor Augmentations in [0,1] gehen
        return (x + noise).clamp(0.0, 1.0)


def _build_transform_train():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply(
            [transforms.RandomAffine(
                degrees=15,
                translate=(0.02, 0.02), 
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


def _load_stats(stats_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            "Erwartet wird dataset_stats_train.json unter images_root.\n"
            "Bitte Preprocessing-Stats generieren (train mean/std)."
        )
    d = json.loads(stats_path.read_text())
    mean = tuple(float(x) for x in d["mean"])
    std = tuple(float(x) for x in d["std"])
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Expected mean/std length 3, got mean={mean}, std={std}")
    return mean, std

def _as_chw_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    x = torch.from_numpy(arr).float()

    if arr.dtype == np.uint8:
        x = x / 255.0
    else:
        if x.numel() > 0 and float(x.max()) > 10.0:
            x = x / 255.0

    # shape -> CHW
    if x.ndim == 2:
        x = x.unsqueeze(0)  
    elif x.ndim == 3:
        # HWC -> CHW
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1)
        # sonst: bereits CxHxW
    else:
        raise ValueError(f"Unsupported npy shape: {tuple(x.shape)}")

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    return x


class To01ForAug(torch.nn.Module):
    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(3, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Heuristik: wenn außerhalb [0,1] -> treat as z-score
        xmin = float(x.min()) if x.numel() > 0 else 0.0
        xmax = float(x.max()) if x.numel() > 0 else 1.0

        if xmin < -1e-3 or xmax > 1.0 + 1e-3:
            std = torch.clamp(self.std, min=self.eps)
            x01 = x * std + self.mean
        else:
            x01 = x

        return x01.clamp(0.0, 1.0)


class NormalizeFrom01(torch.nn.Module):
    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(3, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.clamp(self.std, min=self.eps)
        return (x - self.mean) / std


class NormalizeIfNeeded(torch.nn.Module):
    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(3, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xmin = float(x.min()) if x.numel() > 0 else 0.0
        xmax = float(x.max()) if x.numel() > 0 else 1.0

        if xmin < -1e-3 or xmax > 1.0 + 1e-3:
            return x

        std = torch.clamp(self.std, min=self.eps)
        return (x - self.mean) / std


def _build_train_pipeline(mean, std):
    return transforms.Compose([
        To01ForAug(mean, std),
        _build_transform_train(),    
        NormalizeFrom01(mean, std),
    ])


def _build_eval_pipeline(mean, std):
    return transforms.Compose([
        NormalizeIfNeeded(mean, std),
    ])


class NpyEmotionFolder(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

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
        x = _as_chw_tensor(arr)

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
      images_mtcnn_cropped/
        dataset_stats_train.json
        npy/train/<emotion>/*.npy
        npy/val/<emotion>/*.npy
        npy/test/<emotion>/*.npy
    """
    images_root = Path(images_root)
    npy_root = images_root / "npy"

    mean, std = _load_stats(images_root / "dataset_stats_train.json")

    train_ds = NpyEmotionFolder(npy_root / "train", transform=_build_train_pipeline(mean, std))
    val_ds   = NpyEmotionFolder(npy_root / "val",   transform=_build_eval_pipeline(mean, std))
    test_ds  = NpyEmotionFolder(npy_root / "test",  transform=_build_eval_pipeline(mean, std))

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
    print("batch mean/std:", float(xb.mean()), float(xb.std()))


if __name__ == "__main__":
    main_quickcheck()
