#!/usr/bin/env python3
# file: vision_lab/testing/download_ckplus.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


# Nur die Emotionen, die genutzt werden sollen
EMOTION_DIR_MAP = {
    "Angry": "anger",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happy": "happiness",
    "Sad": "sadness",
    "Surprise": "surprise",
}

IGNORED_DIRS = {"Neutral", "neutral"}


def find_ck_root(download_path: Path) -> Path:
    """
    Findet den CK+ Root-Ordner innerhalb des KaggleHub-Cache.
    Erwartet Strukturen wie:
      CK+/Anger/...
      Anger/...
    """
    ck = download_path / "CK+"
    if ck.exists() and ck.is_dir():
        return ck

    if any((download_path / k).is_dir() for k in EMOTION_DIR_MAP.keys()):
        return download_path

    for d in download_path.rglob("*"):
        if d.is_dir() and any((d / k).is_dir() for k in EMOTION_DIR_MAP.keys()):
            return d

    raise FileNotFoundError(
        f"Could not locate CK+ root inside: {download_path}\n"
        f"Expected CK+/Anger/... or Anger/... etc."
    )


def copy_dataset(src_root: Path, dst_root: Path, *, overwrite: bool) -> None:
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"Destination already exists: {dst_root} (use --overwrite)")

    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)


def lowercase_emotion_dirs(dst_root: Path) -> None:
    # 1) Neutral explizit entfernen
    for name in IGNORED_DIRS:
        p = dst_root / name
        if p.exists() and p.is_dir():
            shutil.rmtree(p)

    # 2) Emotion-Ordner umbenennen (Anger -> anger, ...)
    for old_name, new_name in EMOTION_DIR_MAP.items():
        src = dst_root / old_name
        if not src.exists() or not src.is_dir():
            continue

        dst = dst_root / new_name
        if dst.exists():
            # Merge-Fall
            for item in src.iterdir():
                target = dst / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(src)
        else:
            src.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Target base directory (default: vision_lab/testing).",
    )
    parser.add_argument(
        "--folder-name",
        type=str,
        default="ckplus",
        help="Subfolder name under out-dir (default: ckplus).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target folder.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    dst_root = out_dir / args.folder_name

    download_str = kagglehub.dataset_download("zhiguocui/ck-dataset")
    download_path = Path(download_str).expanduser().resolve()

    src_root = find_ck_root(download_path)
    copy_dataset(src_root, dst_root, overwrite=args.overwrite)
    lowercase_emotion_dirs(dst_root)

    print(f"Downloaded cache : {download_path}")
    print(f"Copied dataset  : {src_root} -> {dst_root}")
    print("Neutral removed, emotion folders normalized to lowercase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
