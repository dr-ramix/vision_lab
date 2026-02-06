from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import hashlib

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STD_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _find_child_dir_ci(parent: Path, wanted: str) -> Path | None:
    w = wanted.lower()
    if not parent.exists():
        return None
    for c in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
        if c.is_dir() and c.name.lower() == w:
            return c
    return None


def _find_first_dir_ci(parent: Path, names: list[str]) -> Path | None:
    for n in names:
        d = _find_child_dir_ci(parent, n)
        if d is not None:
            return d
    return None

def normalize_label(name: str) -> str | None:
    key = name.strip().lower()
    synonyms = {
        # anger
        "Anger": "anger",
        "anger": "anger",
        "Angry": "anger",
        "angry": "anger",
        "6": "anger",
        # disgust
        "disgust": "disgust",
        "Disgust": "disgust",
        "disgusted": "disgust",
        "Disgusted": "disgust",
        "3": "disgust",
        # fear
        "Fear": "fear",
        "fear": "fear",
        "2": "fear",
        # happiness
        "Happy": "happiness",
        "happy": "happiness",
        "Happiness": "happniness",
        "happiness": "happiness",
        "4": "happiness",
        # sadness
        "Sad": "sadness",
        "sad": "sadness",
        "Sadness": "sadness",
        "sadness": "sadness",
        "5": "sadness",
        # surprise
        "Surprised": "surprise",
        "surprised": "surprise",
        "Surprise": "surprise",
        "surprise": "surprise",
        "Suprise": "surprise",
        "suprise": "surprise",
        "1": "surprise",
        # drop / ignore
        "neutral": None,
        "Neutral": None,
        "7": None,
        "contempt": None,
        "Contempt": None
    }
    return synonyms.get(key, key if key in STD_CLASSES else None)


def file_md5(path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache[path] = digest
    return digest


def detect_layout(dataset_root: Path):
    train_dir = _find_first_dir_ci(dataset_root, ["train", "training"])
    test_dir = _find_first_dir_ci(dataset_root, ["test", "testing"])
    val_dir = _find_first_dir_ci(dataset_root, ["validation", "val", "valid"])

    if train_dir is not None and test_dir is not None:
        return ("split", train_dir, test_dir, val_dir)

    return ("unsplit", dataset_root, None, None)


def collect_class_images(root: Path):
    out = {c: [] for c in STD_CLASSES}
    if not root or not root.exists():
        return out

    for sub in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not sub.is_dir():
            continue
        mapped = normalize_label(sub.name)
        if mapped is None or mapped not in out:
            continue

        files = [p for p in sub.rglob("*") if is_img(p)]

        files.sort(key=lambda p: (p.name.lower(), str(p).lower()))
        out[mapped].extend(files)

    return out


def val_from_train_deterministic(
    train_by_class: dict[str, list[Path]],
    val_frac: float,
    content_hash_cache: dict[Path, str],
):
    out = {"train": {}, "val": {}}
    for cls, paths in train_by_class.items():
        paths = list(paths)
        paths.sort(key=lambda p: (file_md5(p, content_hash_cache), p.name.lower()))
        n = len(paths)
        n_val = int(round(n * val_frac))
        out["val"][cls] = paths[:n_val]
        out["train"][cls] = paths[n_val:]
    return out


def stratified_split_deterministic(
    paths_by_class: dict[str, list[Path]],
    ratios: tuple[float, float, float],
    content_hash_cache: dict[Path, str],
):
    split = {"train": {}, "val": {}, "test": {}}
    r_train, r_val, r_test = ratios

    for cls, paths in paths_by_class.items():
        paths = list(paths)
        paths.sort(key=lambda p: (file_md5(p, content_hash_cache), p.name.lower()))
        n = len(paths)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        n_test = max(0, n - n_train - n_val)

        split["train"][cls] = paths[:n_train]
        split["val"][cls] = paths[n_train : n_train + n_val]
        split["test"][cls] = paths[n_train + n_val : n_train + n_val + n_test]

    return split


def stable_name(prefix: str, src: Path, content_hash_cache: dict[Path, str]):
    """
    Stable unique filename so multiple datasets can coexist AND be identical across machines.
    Uses dataset prefix + content hash + original basename.
    """
    h = file_md5(src, content_hash_cache)[:10]
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


def write_split_images(
    split_dict: dict,
    out_raw: Path,
    mode: str,
    on_conflict: str,
    prefix: str,
    content_hash_cache: dict[Path, str],
):
    written = 0
    for split_name in ["train", "val", "test"]:
        if split_name not in split_dict:
            continue
        cls_map = split_dict[split_name]
        for cls in STD_CLASSES:
            paths = cls_map.get(cls, [])
            for src in paths:
                dst = out_raw / split_name / cls / stable_name(prefix, src, content_hash_cache)
                if copy_or_link(src, dst, mode, on_conflict):
                    written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Kept for manifest/backward compatibility (not used for split).")
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
        help="Val share from train when no validation folder exists (e.g., RAFDB).",
    )
    ap.add_argument(
        "--unsplit-ratio",
        nargs=3,
        type=float,
        default=[0.6, 0.2, 0.2],
        help="train val test ratios when dataset has no split",
    )
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    ap.add_argument(
        "--on-conflict",
        choices=["skip", "overwrite"],
        default="skip",
        help="If output file exists: skip or overwrite",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]  # main/
    dataset_root = project_root / "src" / "fer" / "dataset"
    sources_root = dataset_root / "sources"
    out_raw = dataset_root / "standardized" / "images_raw"
    splits_root = dataset_root / "splits"

    safe_mkdir(sources_root)
    safe_mkdir(out_raw)
    safe_mkdir(splits_root)

    datasets = ["rafdb", "ferplus"]

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "unsplit_ratio": args.unsplit_ratio,
        "mode": args.mode,
        "on_conflict": args.on_conflict,
        "datasets": {},
        "deterministic_val": "content_md5_sorted_per_class",
    }

    # ensure base split/class dirs exist
    for split in ["train", "val", "test"]:
        for cls in STD_CLASSES:
            safe_mkdir(out_raw / split / cls)

    # cache for content hashes (important for speed)
    content_hash_cache: dict[Path, str] = {}

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

                written += write_split_images(
                    {"train": train_by_class},
                    out_raw,
                    args.mode,
                    args.on_conflict,
                    ds_name,
                    content_hash_cache,
                )
                written += write_split_images(
                    {"val": val_by_class},
                    out_raw,
                    args.mode,
                    args.on_conflict,
                    ds_name,
                    content_hash_cache,
                )

                train_counts = {c: len(train_by_class[c]) for c in STD_CLASSES}
                val_counts = {c: len(val_by_class[c]) for c in STD_CLASSES}
            else:
                # RAFDB: no validation -> carve out val_frac deterministically from train per class
                tv = val_from_train_deterministic(
                    train_by_class,
                    val_frac=args.val_frac,
                    content_hash_cache=content_hash_cache,
                )

                written += write_split_images(
                    {"train": tv["train"], "val": tv["val"]},
                    out_raw,
                    args.mode,
                    args.on_conflict,
                    ds_name,
                    content_hash_cache,
                )

                train_counts = {c: len(tv["train"][c]) for c in STD_CLASSES}
                val_counts = {c: len(tv["val"][c]) for c in STD_CLASSES}

            # test always full
            written += write_split_images(
                {"test": test_by_class},
                out_raw,
                args.mode,
                args.on_conflict,
                ds_name,
                content_hash_cache,
            )

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
            # unsplit dataset: split deterministically into train/val/test via ratios
            all_by_class = collect_class_images(ds_root)
            ratios = tuple(args.unsplit_ratio)
            split = stratified_split_deterministic(
                all_by_class,
                ratios=ratios,
                content_hash_cache=content_hash_cache,
            )
            written = write_split_images(
                split,
                out_raw,
                args.mode,
                args.on_conflict,
                ds_name,
                content_hash_cache,
            )

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
