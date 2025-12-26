# setx KAGGLE_API_TOKEN <DEIN API TOKEN>
# pip install kagglehub

"""
download_sources.py

Lädt FER2013 und AffectNet über kagglehub herunter
und kopiert sie nach:

vision_lab/src/fer/dataset/sources/
  ├── fer2013/
  └── affectnet/

Voraussetzungen:
- pip install kagglehub
- kaggle API Key eingerichtet (~/.kaggle/kaggle.json)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import kagglehub


@dataclass
class DatasetSpec:
    kaggle_id: str          # z.B. "mstjebashazida/affectnet"
    target_name: str        # z.B. "affectnet"


def _is_dataset_root(p: Path) -> bool:
    """Ein 'Root' ist ein Ordner, der train/ und test/ enthält (Datei labels.csv optional)."""
    return p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()


def _find_best_root(download_path: Path) -> Path:
    """
    Sucht rekursiv nach einem Ordner, der train/ und test/ enthält.
    Nimmt den 'tiefsten' Treffer (meistens der innere archive-Ordner).
    """
    candidates = []
    for d in download_path.rglob("*"):
        if _is_dataset_root(d):
            candidates.append(d)

    if _is_dataset_root(download_path):
        candidates.append(download_path)

    if not candidates:
        raise FileNotFoundError(
            f"Konnte keinen Dataset-Root mit train/ und test/ unter {download_path} finden."
        )

    # tiefster Pfad = meistens der richtige (z.B. .../archive (3))
    candidates.sort(key=lambda x: len(x.parts), reverse=True)
    return candidates[0]


def _sync_dir(src: Path, dst: Path, overwrite: bool = True) -> None:
    """
    Kopiert src -> dst.
    Wenn overwrite=True, löscht dst vorher komplett.
    """
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _copy_required_layout(dataset_root: Path, dst_root: Path) -> None:
    """
    Kopiert nur train/, test/ und optional labels.csv nach dst_root.
    Dadurch entfernen wir "doppelte Ordnerstruktur" zuverlässig.
    """
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # train + test sind Pflicht
    shutil.copytree(dataset_root / "train", dst_root / "train")
    shutil.copytree(dataset_root / "test", dst_root / "test")

    # labels.csv optional
    lab = dataset_root / "labels.csv"
    if lab.exists():
        shutil.copy2(lab, dst_root / "labels.csv")


def download_and_place(spec: DatasetSpec, sources_dir: Path) -> Path:
    """
    Lädt Dataset via kagglehub, findet den echten Root (train/test),
    und legt es flach in sources/<target_name>/ ab.
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


def main():
    # Passe das an, falls dein Script woanders liegt:
    # Script liegt in main/scripts/ -> parents[1] = .../main
    project_main = Path(__file__).resolve().parents[1]
    sources_dir = project_main / "src" / "fer" / "dataset" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("msambare/fer2013", "fer2013"),
        DatasetSpec("mstjebashazida/affectnet", "affectnet"),
    ]

    for spec in specs:
        download_and_place(spec, sources_dir)

    print("\nDone. Sources ready.")


if __name__ == "__main__":
    main()
