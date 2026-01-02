"""
download_sources.py

Downloads FERPlus, AffectNet, and RAF-DB via kagglehub and copies them into:

vision_lab/main/src/fer/dataset/sources/
  ├── ferplus/
  ├── affectnet/
  └── raf-db/

Robust features:
- Finds dataset root recursively (even if Kaggle wraps files in "archive", "archive (3)", etc.)
- Case-insensitive detection of Train/Test/Validation vs train/test/validation
- Copies only train/, test/ and optional validation/ and optional labels.csv into a flat, consistent layout

Requirements:
- pip install kagglehub
- Kaggle credentials set up (e.g. ~/.kaggle/kaggle.json or env token supported by kagglehub)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import kagglehub


@dataclass(frozen=True)
class DatasetSpec:
    kaggle_id: str
    target_name: str


def _find_child_dir_ci(parent: Path, wanted_name: str) -> Optional[Path]:
    """Find direct child directory by name, case-insensitive."""
    wanted = wanted_name.lower()
    try:
        for c in parent.iterdir():
            if c.is_dir() and c.name.lower() == wanted:
                return c
    except FileNotFoundError:
        return None
    return None


def _find_child_file_ci(parent: Path, wanted_name: str) -> Optional[Path]:
    """Find direct child file by name, case-insensitive."""
    wanted = wanted_name.lower()
    try:
        for c in parent.iterdir():
            if c.is_file() and c.name.lower() == wanted:
                return c
    except FileNotFoundError:
        return None
    return None


def _find_first_existing_dir_ci(parent: Path, names: list[str]) -> Optional[Path]:
    """Try multiple directory names (case-insensitive) and return the first match."""
    for name in names:
        d = _find_child_dir_ci(parent, name)
        if d is not None:
            return d
    return None


def _is_dataset_root(p: Path) -> bool:
    """
    Root folder is any directory that contains train/ and test/ (case-insensitive).
    validation/ is optional.
    """
    if not p.is_dir():
        return False
    return _find_child_dir_ci(p, "train") is not None and _find_child_dir_ci(p, "test") is not None


def _find_best_root(download_path: Path) -> Path:
    """
    Recursively search for a folder containing train/ and test/ (case-insensitive).
    Choose the deepest match (most nested) to handle Kaggle "archive" wrappers.
    """
    candidates: list[Path] = []

    if _is_dataset_root(download_path):
        candidates.append(download_path)

    for d in download_path.rglob("*"):
        if d.is_dir() and _is_dataset_root(d):
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset root with train/ and test/ under: {download_path}\n"
            f"Tip: inspect the downloaded folder structure and update root detection if needed."
        )

    candidates.sort(key=lambda x: len(x.parts), reverse=True)
    return candidates[0]


def _resolve_train_test(dataset_root: Path) -> Tuple[Path, Path]:
    """Return the actual train and test directories under dataset_root (case-insensitive)."""
    train_dir = _find_child_dir_ci(dataset_root, "train")
    test_dir = _find_child_dir_ci(dataset_root, "test")
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(f"Dataset root missing train/test: {dataset_root}")
    return train_dir, test_dir


def _resolve_validation(dataset_root: Path) -> Optional[Path]:
    """
    Return optional validation directory (case-insensitive).
    Supports common names: validation/, val/, valid/
    """
    return _find_first_existing_dir_ci(dataset_root, ["validation", "val", "valid"])


def _copy_required_layout(dataset_root: Path, dst_root: Path) -> None:
    """
    Copy ONLY:
      - train/  -> dst_root/train
      - test/   -> dst_root/test
      - optional validation/ (validation|val|valid, any casing) -> dst_root/validation
      - optional labels.csv (any casing) -> dst_root/labels.csv

    This ensures consistent output layout regardless of source packaging.
    """
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = _resolve_train_test(dataset_root)

    shutil.copytree(train_dir, dst_root / "train")
    shutil.copytree(test_dir, dst_root / "test")

    val_dir = _resolve_validation(dataset_root)
    if val_dir is not None:
        shutil.copytree(val_dir, dst_root / "validation")

    labels = _find_child_file_ci(dataset_root, "labels.csv")
    if labels is not None:
        shutil.copy2(labels, dst_root / "labels.csv")


def download_and_place(spec: DatasetSpec, sources_dir: Path) -> Path:
    """
    Download dataset via kagglehub, detect true dataset root, and place into sources/<target_name>/.
    """
    print(f"\n=== Download: {spec.kaggle_id} ===")
    dl_path_str = kagglehub.dataset_download(spec.kaggle_id)
    dl_path = Path(dl_path_str)

    print(f"Downloaded to: {dl_path}")

    dataset_root = _find_best_root(dl_path)
    print(f"Detected dataset root: {dataset_root}")

    target = sources_dir / spec.target_name
    _copy_required_layout(dataset_root, target)

    print(f"Placed into: {target}")
    return target


def main() -> None:
    project_main = Path(__file__).resolve().parents[1]
    sources_dir = project_main / "src" / "fer" / "dataset" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("arnabkumarroy02/ferplus", "ferplus"),
        DatasetSpec("mstjebashazida/affectnet", "affectnet"),
        DatasetSpec("shuvoalok/raf-db-dataset", "raf-db"),
    ]

    for spec in specs:
        download_and_place(spec, sources_dir)

    print("\nDone. Sources ready.")


if __name__ == "__main__":
    main()
