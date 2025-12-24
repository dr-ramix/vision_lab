from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def _build_transform_train():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        # leichter Gaussian Blur (Motion-Blur-Ersatz)
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.15,
        ),

        # Kontrast-Jitter:
        # contrast=0.25 -> Faktor in [0.75, 1.25]
        transforms.RandomApply(
            [transforms.ColorJitter(contrast=0.25)],
            p=0.30,
        ),

        transforms.ToTensor(),  # -> (C,H,W), float32, [0,1]
    ])


def _build_transform_eval():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def _remap_dataset_labels(ds: datasets.ImageFolder) -> None:
    """
    Erzwingt CLASS_TO_IDX für ein ImageFolder-Dataset.
    Überschreibt:
      - class_to_idx
      - classes
      - samples / targets
    """
    # Original mapping (alphabetisch)
    old_class_to_idx = ds.class_to_idx

    # Sanity-Check
    missing = set(CLASS_TO_IDX.keys()) - set(old_class_to_idx.keys())
    if missing:
        raise ValueError(f"Dataset missing classes: {missing}")

    # Neue Attribute setzen
    ds.class_to_idx = CLASS_TO_IDX
    ds.classes = CLASS_ORDER

    # samples: List[(path, old_idx)] -> (path, new_idx)
    new_samples = []
    for path, old_idx in ds.samples:
        class_name = next(k for k, v in old_class_to_idx.items() if v == old_idx)
        new_idx = CLASS_TO_IDX[class_name]
        new_samples.append((path, new_idx))

    ds.samples = new_samples
    ds.targets = [s[1] for s in new_samples]


def build_dataloaders(
    images_root: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoaders:
    """
    images_root muss zeigen auf:
      images_mtcnn_cropped_norm/
        train/<class>/*
        val/<class>/*
        test/<class>/*
    """

    images_root = Path(images_root)

    train_tf = _build_transform_train()
    eval_tf = _build_transform_eval()

    train_ds = datasets.ImageFolder(str(images_root / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(images_root / "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(str(images_root / "test"),  transform=eval_tf)

    #  ERZWINGE feste Klassen-IDs
    _remap_dataset_labels(train_ds)
    _remap_dataset_labels(val_ds)
    _remap_dataset_labels(test_ds)

    pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_to_idx=CLASS_TO_IDX,
    )


# --------------------------------------------------
# Quick Sanity Check
# --------------------------------------------------
def main_quickcheck():
    project_root = Path(__file__).resolve().parents[3]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    dls = build_dataloaders(images_root, batch_size=8, num_workers=0)

    xb, yb = next(iter(dls.train))
    print("class_to_idx:", dls.class_to_idx)
    print("labels in batch:", yb.tolist())
    print("input shape:", xb.shape)


if __name__ == "__main__":
    main_quickcheck()
