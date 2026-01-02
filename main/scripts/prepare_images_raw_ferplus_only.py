from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import shutil

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard-Klassen (genau deine 6)
STD_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# FER2013 emotion mapping (klassisch: 0..6 = anger, disgust, fear, happiness, sadness, surprise, neutral)
FER2013_ID_TO_LABEL = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: None,  # neutral -> drop
}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def normalize_label(name: str) -> str | None:
    key = name.strip().lower()
    synonyms = {
        "Anger": "anger",
        "angry": "anger",
        "anger": "anger",
        "Disgust": "disgust",
        "disgust": "disgust",
        "Fear": "fear",
        "fear": "fear",
        "Happy": "happiness",
        "happy": "happiness",
        "Happiness": "happiness",
        "happiness": "happiness",
        "Sad": "sadness",
        "sad": "sadness",
        "Sadness": "sadness",
        "sadness": "sadness",
        "Surprised": "surprise",
        "surprised": "surprise",
        "Surprise": "surprise",
        "surprise": "surprise",
        "Neutral": None,
        "neutral": None,
        "Contempt": None,
        "contempt": None,
    }
    return synonyms.get(key, key if key in STD_CLASSES else None)


def detect_layout(dataset_root: Path):
    """Return ("split", train_dir, test_dir) or ("unsplit", root)."""
    for a, b in [("train", "test"), ("Training", "Testing")]:
        t = dataset_root / a
        s = dataset_root / b
        if t.exists() and s.exists():
            return ("split", t, s)
    return ("unsplit", dataset_root, None)


def collect_class_images(root: Path):
    """
    Expects root/<label>/**.jpg ...
    Returns dict std_label -> [paths]
    """
    out = {c: [] for c in STD_CLASSES}
    if not root.exists():
        return out

    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        mapped = normalize_label(sub.name)
        if mapped is None or mapped not in out:
            continue
        for p in sub.rglob("*"):
            if is_img(p):
                out[mapped].append(p)

    return out


def val_from_train(train_by_class: dict[str, list[Path]], val_frac: float, seed: int):
    rng = random.Random(seed)
    out = {"train": {}, "val": {}}
    for cls, paths in train_by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_val = int(round(n * val_frac))
        out["val"][cls] = paths[:n_val]
        out["train"][cls] = paths[n_val:]
    return out


def stratified_split(paths_by_class: dict[str, list[Path]], ratios: tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    split = {"train": {}, "val": {}, "test": {}}

    for cls, paths in paths_by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        n_test = max(0, n - n_train - n_val)

        split["train"][cls] = paths[:n_train]
        split["val"][cls] = paths[n_train:n_train + n_val]
        split["test"][cls] = paths[n_train + n_val:n_train + n_val + n_test]

    return split


def stable_name(prefix: str, src: Path):
    """
    Create stable unique filename.
    Uses dataset prefix + hash of full path to avoid collisions.
    """
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{h}__{src.name}"


def copy_or_link(src: Path, dst: Path, mode: str, on_conflict: str):
    safe_mkdir(dst.parent)

    if dst.exists():
        if on_conflict == "skip":
            return False
        if on_conflict == "overwrite":
            dst.unlink()
        else:
            raise ValueError("on_conflict must be skip|overwrite")

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError("mode must be copy|hardlink")
    return True


def write_split_images(split_dict: dict, out_raw: Path, mode: str, on_conflict: str, prefix: str):
    written = 0
    for split_name, cls_map in split_dict.items():
        for cls, paths in cls_map.items():
            for src in paths:
                dst = out_raw / split_name / cls / stable_name(prefix, src)
                if copy_or_link(src, dst, mode, on_conflict):
                    written += 1
    return written


def find_fer2013_csv(ds_root: Path) -> Path | None:
    for name in ["fer2013.csv", "FER2013.csv", "icml_face_data.csv"]:
        p = ds_root / name
        if p.exists():
            return p
    for p in ds_root.rglob("*.csv"):
        if p.name.lower() == "fer2013.csv":
            return p
    return None


def ensure_cv2_numpy():
    if np is None:
        raise RuntimeError("numpy not available. Install numpy.")
    if cv2 is None:
        raise RuntimeError("opencv-python not available. Install opencv-python to write FER2013 images.")


def fer2013_usage_to_split(usage: str) -> str | None:
    u = usage.strip().lower()
    if "train" in u:
        return "train"
    if "publictest" in u or "public test" in u:
        return "test"
    if "privatetest" in u or "private test" in u:
        return "test"
    return None


def process_fer2013_csv(csv_path: Path, out_raw: Path, val_frac: float, seed: int, prefix: str, on_conflict: str):
    """
    Writes FER2013 images to out_raw/{train,val,test}/{class}/...
    Strategy:
      - Use Usage column for train/test
      - From train portion, carve out val_frac per class (stratified) using seed
      - Only 6 emotions, drop neutral
    """
    ensure_cv2_numpy()
    rng = random.Random(seed)

    train_rows = {c: [] for c in STD_CLASSES}
    test_rows = {c: [] for c in STD_CLASSES}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels", "Usage"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"FER2013 CSV must have columns {required}, got {reader.fieldnames}")

        for idx, row in enumerate(reader):
            try:
                emo = int(row["emotion"])
            except Exception:
                continue
            cls = FER2013_ID_TO_LABEL.get(emo, None)
            if cls is None or cls not in STD_CLASSES:
                continue

            split = fer2013_usage_to_split(row["Usage"])
            if split == "train":
                train_rows[cls].append((idx, row))
            elif split == "test":
                test_rows[cls].append((idx, row))

    val_rows = {c: [] for c in STD_CLASSES}
    train_keep = {c: [] for c in STD_CLASSES}

    for cls, rows in train_rows.items():
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        n_val = int(round(n * val_frac))
        val_rows[cls] = rows[:n_val]
        train_keep[cls] = rows[n_val:]

    def write_rows(split_name: str, rows_by_class: dict[str, list[tuple[int, dict]]]):
        written = 0
        for cls, rows in rows_by_class.items():
            for idx, row in rows:
                pixels = row["pixels"].strip().split()
                arr = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                img = arr  # grayscale PNG

                name = f"{prefix}__{idx:06d}.png"
                dst = out_raw / split_name / cls / name
                safe_mkdir(dst.parent)

                if dst.exists():
                    if on_conflict == "skip":
                        continue
                    if on_conflict == "overwrite":
                        dst.unlink()
                cv2.imwrite(str(dst), img)
                written += 1
        return written

    written = 0
    written += write_rows("train", train_keep)
    written += write_rows("val", val_rows)
    written += write_rows("test", test_rows)

    return {
        "layout": "fer2013_csv",
        "counts": {
            "train": {c: len(train_keep[c]) for c in STD_CLASSES},
            "val": {c: len(val_rows[c]) for c in STD_CLASSES},
            "test": {c: len(test_rows[c]) for c in STD_CLASSES},
        },
        "written": written,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10, help="Val share from train when train/test exists")
    ap.add_argument("--unsplit-ratio", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train val test ratios when dataset has no split")
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    ap.add_argument("--on-conflict", choices=["skip", "overwrite"], default="skip",
                    help="If output file exists: skip or overwrite")
    args = ap.parse_args()

    # main/scripts/...  -> parents[1] == main/
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "src" / "fer" / "dataset"

    sources_root = dataset_root / "sources"
    ds_name = "ferplus"
    ds_root = sources_root / ds_name

    out_raw = dataset_root / "standardized" / "ferplus" / "ferplus_raw"
    splits_root = dataset_root / "splits"

    safe_mkdir(out_raw)
    safe_mkdir(splits_root)

    if not ds_root.exists():
        raise SystemExit(f"[error] Dataset not found: {ds_root}")

    # ensure base split/class dirs exist
    for split in ["train", "val", "test"]:
        for cls in STD_CLASSES:
            safe_mkdir(out_raw / split / cls)

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "unsplit_ratio": args.unsplit_ratio,
        "mode": args.mode,
        "on_conflict": args.on_conflict,
        "datasets": {},
    }

    print(f"\n=== Dataset (only): {ds_name} ===")
    print(f"source: {ds_root}")

    # Keep old FER2013 CSV handling intact (but now it will simply not trigger for ferplus)
    fer_csv = find_fer2013_csv(ds_root)
    if fer_csv is not None:
        info = process_fer2013_csv(
            csv_path=fer_csv,
            out_raw=out_raw,
            val_frac=args.val_frac,
            seed=args.seed,
            prefix=ds_name,
            on_conflict=args.on_conflict,
        )
        manifest["datasets"][ds_name] = info
        print(f"  -> FER2013 CSV detected: {fer_csv.name}")
        print(f"  -> written: {info['written']} images")
    else:
        kind, train_dir, test_dir = detect_layout(ds_root)

        if kind == "split":
            train_by_class = collect_class_images(train_dir)
            test_by_class = collect_class_images(test_dir)

            tv = val_from_train(train_by_class, val_frac=args.val_frac, seed=args.seed)
            written = 0
            written += write_split_images(
                {"train": tv["train"], "val": tv["val"]},
                out_raw,
                args.mode,
                args.on_conflict,
                ds_name,
            )
            written += write_split_images(
                {"test": test_by_class},
                out_raw,
                args.mode,
                args.on_conflict,
                ds_name,
            )

            manifest["datasets"][ds_name] = {
                "layout": "train_test_dirs",
                "counts": {
                    "train": {c: len(tv["train"][c]) for c in STD_CLASSES},
                    "val": {c: len(tv["val"][c]) for c in STD_CLASSES},
                    "test": {c: len(test_by_class[c]) for c in STD_CLASSES},
                },
                "written": written,
            }
            print(f"  -> layout: train/test dirs | written: {written}")
        else:
            all_by_class = collect_class_images(ds_root)
            ratios = tuple(args.unsplit_ratio)
            split = stratified_split(all_by_class, ratios=ratios, seed=args.seed)
            written = write_split_images(split, out_raw, args.mode, args.on_conflict, ds_name)

            manifest["datasets"][ds_name] = {
                "layout": "unsplit_class_dirs",
                "counts": {
                    "train": {c: len(split['train'][c]) for c in STD_CLASSES},
                    "val": {c: len(split['val'][c]) for c in STD_CLASSES},
                    "test": {c: len(split['test'][c]) for c in STD_CLASSES},
                },
                "written": written,
            }
            print(f"  -> layout: unsplit | written: {written}")

    manifest_path = dataset_root / "standardized" / "ferplus" / "ferplus_raw" / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"images_raw: {out_raw}")
    print(f"manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
