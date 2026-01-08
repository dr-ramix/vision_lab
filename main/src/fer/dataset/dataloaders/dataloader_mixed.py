# file: vision_lab/main/src/fer/dataset/dataloaders/dataloader_mixed.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# Class mapping (fixed order)
# ============================================================
CLASS_ORDER: List[str] = [
    "anger",      # 0
    "disgust",    # 1
    "fear",       # 2
    "happiness",  # 3
    "sadness",    # 4
    "surprise",   # 5
]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}


# ============================================================
# Small container to match other loaders (dls.train/val/test)
# ============================================================
@dataclass
class _DLS:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: Dict[str, int]


# ============================================================
# Import train augmentations from dataloader_grey (re-usable)
# ============================================================
from fer.dataset.dataloaders.dataloader_grey import build_train_augmentations


def _read_stats(stats_json: Path) -> Tuple[List[float], List[float], Tuple[int, int]]:
    obj = json.loads(stats_json.read_text(encoding="utf-8"))
    mean = [float(x) for x in obj["mean"]]
    std = [float(x) for x in obj["std"]]
    ts = obj.get("target_size", [64, 64])
    target_size = (int(ts[0]), int(ts[1]))
    return mean, std, target_size


# ============================================================
# resolve paths given images_root = ".../standardized"
# ============================================================
def _resolve_mixed_paths(standardized_root: Path) -> Tuple[Path, Path]:
    """
    User contract:
      images_root passed in from registry/settings is:
        .../main/src/fer/dataset/standardized

    We must find:
      npy_root   = standardized_root/only_mtcnn_cropped/color_and_grey/npy
      stats_json = standardized_root/only_mtcnn_cropped/color_and_grey/dataset_stats_train.json

    Also supports the case where someone accidentally passes already the npy_root.
    """
    standardized_root = Path(standardized_root)

    # case A: caller already passes ".../only_mtcnn_cropped/color_and_grey/npy"
    if standardized_root.name == "npy" and standardized_root.parent.name == "color_and_grey":
        npy_root = standardized_root
        stats_json = standardized_root.parent / "dataset_stats_train.json"
        return npy_root, stats_json

    # case B: caller passes ".../standardized"
    npy_root = standardized_root / "only_mtcnn_cropped" / "color_and_grey" / "npy"
    stats_json = standardized_root / "only_mtcnn_cropped" / "color_and_grey" / "dataset_stats_train.json"
    return npy_root, stats_json


# ============================================================
# Dataset (same logic as grey)
# ============================================================
class NpyFolderDataset(Dataset):
    def __init__(
        self,
        split_root: Path,
        class_to_idx: Dict[str, int],
        *,
        train: bool,
        mean: List[float],
        std: List[float],
        augment: Optional[torch.nn.Module] = None,
    ):
        self.split_root = Path(split_root)
        self.class_to_idx = dict(class_to_idx)
        self.train = bool(train)

        self.augment = augment if train else None
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.samples: List[Tuple[Path, int]] = []
        for cls in CLASS_ORDER:
            cls_dir = self.split_root / cls
            if not cls_dir.exists():
                continue
            y = self.class_to_idx[cls]
            for p in sorted(cls_dir.rglob("*.npy")):
                if p.is_file():
                    self.samples.append((p, y))

        if not self.samples:
            raise FileNotFoundError(f"No .npy files found under: {self.split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_chw_float01(x: np.ndarray) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, :, None]
        if x.ndim != 3:
            raise ValueError(f"Expected 2D/3D npy, got shape={x.shape}")

        # HWC -> CHW (if last dim looks like channels)
        if x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = np.transpose(x, (2, 0, 1))

        if x.shape[0] not in (1, 3):
            raise ValueError(f"Expected channels=1 or 3, got shape={x.shape}")

        xt = torch.from_numpy(x).float()

        if xt.max().item() > 1.5:
            xt = xt / 255.0

        # IMPORTANT for mixed: if 1-channel, repeat to 3 because stats are 3-channel
        if xt.shape[0] == 1:
            xt = xt.repeat(3, 1, 1)

        return xt.clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p, y = self.samples[idx]
        x = np.load(p)
        x = self._to_chw_float01(x)  # CHW, float, [0,1]

        if self.augment is not None:
            x = self.augment(x)

        x = self.normalize(x)
        return x, y


# ============================================================
# Public builder (matches registry expectation)
# ============================================================
def build_dataloaders(
    *,
    images_root: Path,              # expected: .../standardized
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    stats_json: Optional[Path] = None,
) -> _DLS:
    """
    images_root (settings.images_root):
      .../main/src/fer/dataset/standardized

    actual data:
      .../standardized/only_mtcnn_cropped/color_and_grey/npy/{train,val,test}/{class}/*.npy

    stats:
      .../standardized/only_mtcnn_cropped/color_and_grey/dataset_stats_train.json
    """
    images_root = Path(images_root)

    npy_root, default_stats = _resolve_mixed_paths(images_root)
    stats_json = Path(default_stats) if stats_json is None else Path(stats_json)

    if not npy_root.exists():
        raise FileNotFoundError(f"Mixed npy root not found: {npy_root}")
    if not stats_json.exists():
        raise FileNotFoundError(f"Stats JSON not found: {stats_json}")

    mean, std, _target_size = _read_stats(stats_json)

    train_aug = build_train_augmentations()

    train_ds = NpyFolderDataset(
        npy_root / "train",
        CLASS_TO_IDX,
        train=True,
        mean=mean,
        std=std,
        augment=train_aug,
    )
    val_ds = NpyFolderDataset(
        npy_root / "val",
        CLASS_TO_IDX,
        train=False,
        mean=mean,
        std=std,
        augment=None,
    )
    test_ds = NpyFolderDataset(
        npy_root / "test",
        CLASS_TO_IDX,
        train=False,
        mean=mean,
        std=std,
        augment=None,
    )

    def _dl(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return _DLS(
        train=_dl(train_ds, shuffle=True),
        val=_dl(val_ds, shuffle=False),
        test=_dl(test_ds, shuffle=False),
        class_to_idx=dict(CLASS_TO_IDX),
    )
