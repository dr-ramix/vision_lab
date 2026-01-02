from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import shutil
import hashlib

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard-Klassen (genau deine 6)
STD_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


# ------------------ basic helpers ------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _find_child_dir_ci(parent: Path, wanted: str) -> Path | None:
    """Find direct child directory by name, case-insensitive (Linux-safe)."""
    w = wanted.lower()
    if not parent.exists():
        return None
    for c in parent.iterdir():
        if c.is_dir() and c.name.lower() == w:
            return c
    return None


def _find_first_dir_ci(parent: Path, names: list[str]) -> Path | None:
    """Try multiple directory names (case-insensitive) and return first match."""
    for n in names:
        d = _find_child_dir_ci(parent, n)
        if d is not None:
            return d
    return None


# ------------------ label mapping (KEEP your synonyms!) ------------------
def normalize_label(name: str) -> str | None:
    key = name.strip().lower()
    synonyms = {
        # anger
        "anger": "anger",
        "angry": "anger",
        "6": "anger",

        # disgust
        "disgust": "disgust",
        "3": "disgust",

        # fear
        "fear": "fear",
        "2": "fear",

        # happiness
        "happy": "happiness",
        "happiness": "happiness",
        "4": "happiness",

        # sadness
        "sad": "sadness",
        "sadness": "sadness",
        "5": "sadness",

        # surprise
        "surprised": "surprise",
        "surprise": "surprise",
        "1": "surprise",

        # drop / ignore
        "neutral": None,
        "7": None,
        "contempt": None,
    }
    return synonyms.get(key, key if key in STD_CLASSES else None)


# ------------------ layout detection (supports validation for FERPlus) ------------------
def detect_layout(dataset_root: Path):
    """
    Return:
      ("split", train_dir, test_dir, val_dir_or_None)
      ("unsplit", root, None, None)

    Detected case-insensitive:
      train: train | training
      test: test | testing
      val: validation | val | valid
    """
    train_dir = _find_first_dir_ci(dataset_root, ["train", "training"])
    test_dir = _find_first_dir_ci(dataset_root, ["test", "testing"])
    val_dir = _find_first_dir_ci(dataset_root, ["validation", "val", "valid"])

    if train_dir is not None and test_dir is not None:
        return ("split", train_dir, test_dir, val_dir)

    return ("unsplit", dataset_root, None, None)


# ------------------ collecting & splitting ------------------
def collect_class_images(root: Path):
    """
    Expects root/<label>/**.jpg ...
    Returns dict std_label -> [paths]
    """
    out = {c: [] for c in STD_CLASSES}
    if not root or not root.exists():
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


# ------------------ writing images_raw ------------------
def stable_name(prefix: str, src: Path):
    """
    Stable unique filename so multiple datasets can coexist.
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


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Val share from train when no validation folder exists (AffectNet/RAF-DB)")
    ap.add_argument("--unsplit-ratio", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train val test ratios when dataset has no split")
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    ap.add_argument("--on-conflict", choices=["skip", "overwrite"], default="skip",
                    help="If output file exists: skip or overwrite")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]  # main/
    dataset_root = project_root / "src" / "fer" / "dataset"
    sources_root = dataset_root / "sources"
    out_raw = dataset_root / "standardized" / "images_raw"
    splits_root = dataset_root / "splits"

    safe_mkdir(sources_root)
    safe_mkdir(out_raw)
    safe_mkdir(splits_root)

    # ONLY these three datasets (as requested)
    datasets = ["affectnet", "raf-db", "ferplus"]

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "unsplit_ratio": args.unsplit_ratio,
        "mode": args.mode,
        "on_conflict": args.on_conflict,
        "datasets": {}
    }

    # ensure base split/class dirs exist
    for split in ["train", "val", "test"]:
        for cls in STD_CLASSES:
            safe_mkdir(out_raw / split / cls)

    for ds_name in datasets:
        ds_root = sources_root / ds_name
        if not ds_root.exists():
            print(f"[skip] dataset not found: {ds_root}")
            continue

        print(f"\n=== Dataset: {ds_name} ===")

        kind, train_dir, test_dir, val_dir = detect_layout(ds_root)

        if kind == "split":
            train_by_class = collect_class_images(train_dir)
            test_by_class = collect_class_images(test_dir)

            written = 0

            # FERPlus: validation exists -> use it fully, DO NOT touch train
            if val_dir is not None:
                val_by_class = collect_class_images(val_dir)

                written += write_split_images({"train": train_by_class}, out_raw, args.mode, args.on_conflict, ds_name)
                written += write_split_images({"val": val_by_class}, out_raw, args.mode, args.on_conflict, ds_name)

                train_counts = {c: len(train_by_class[c]) for c in STD_CLASSES}
                val_counts = {c: len(val_by_class[c]) for c in STD_CLASSES}
            else:
                # AffectNet / RAF-DB: no validation -> carve out val_frac from train per class
                tv = val_from_train(train_by_class, val_frac=args.val_frac, seed=args.seed)

                written += write_split_images({"train": tv["train"], "val": tv["val"]},
                                             out_raw, args.mode, args.on_conflict, ds_name)

                train_counts = {c: len(tv["train"][c]) for c in STD_CLASSES}
                val_counts = {c: len(tv["val"][c]) for c in STD_CLASSES}

            # test always full
            written += write_split_images({"test": test_by_class}, out_raw, args.mode, args.on_conflict, ds_name)

            manifest["datasets"][ds_name] = {
                "layout": "train_test_dirs(+optional_validation)",
                "has_validation_dir": val_dir is not None,
                "counts": {
                    "train": train_counts,
                    "val": val_counts,
                    "test": {c: len(test_by_class[c]) for c in STD_CLASSES},
                },
                "written": written,
            }
            print(f"  -> written: {written} | validation_dir: {val_dir is not None}")

        else:
            # unsplit dataset: split into train/val/test via ratios
            all_by_class = collect_class_images(ds_root)
            ratios = tuple(args.unsplit_ratio)
            split = stratified_split(all_by_class, ratios=ratios, seed=args.seed)
            written = write_split_images(split, out_raw, args.mode, args.on_conflict, ds_name)

            manifest["datasets"][ds_name] = {
                "layout": "unsplit_class_dirs",
                "counts": {
                    "train": {c: len(split["train"][c]) for c in STD_CLASSES},
                    "val": {c: len(split["val"][c]) for c in STD_CLASSES},
                    "test": {c: len(split["test"][c]) for c in STD_CLASSES},
                },
                "written": written,
            }
            print(f"  -> layout: unsplit | written: {written}")

    manifest_path = splits_root / "images_raw_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"images_raw: {out_raw}")
    print(f"manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
