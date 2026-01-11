# file: vision_lab/main/scripts/compute_stats_images_mtcnn_cropped_norm.py
# Computes mean/std over TRAIN split only for:
#   .../main/src/fer/dataset/standardized/images_mtcnn_cropped_norm/npy/train/<class>/*.npy
#
# Writes:
#   .../main/src/fer/dataset/standardized/images_mtcnn_cropped_norm/dataset_stats_train.json

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
SPLIT = "train"
EXT = ".npy"


def compute_mean_std_from_npy(train_root_npy: Path) -> tuple[list[float], list[float]]:
    """
    Expects each .npy to be:
      - float32/float64 in [0,1] (preferred) OR 0..255 (will be scaled)
      - shape either HWC (H,W,3) or CHW (3,H,W)
    Computes per-channel mean/std over all pixels.
    """
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for cls in CLASSES:
        cls_dir = train_root_npy / cls
        if not cls_dir.exists():
            print(f"[skip] missing class dir: {cls_dir}")
            continue

        files = [p for p in cls_dir.rglob(f"*{EXT}") if p.is_file()]
        print(f"[npy] {SPLIT}/{cls} | files: {len(files)}")

        for p in files:
            x = np.load(p)

            if x.ndim != 3:
                raise ValueError(
                    f"Invalid array shape in {p}: expected 3D array (HWC or CHW), got {x.shape}"
                )

            # CHW -> HWC
            if x.shape[0] == 3 and x.shape[-1] != 3:
                x = np.transpose(x, (1, 2, 0))
            elif x.shape[-1] != 3:
                raise ValueError(f"Expected 3 channels in {p}, got shape {x.shape}")

            x = x.astype(np.float64, copy=False)

            # safety: if someone saved 0..255
            if x.max() > 1.5:
                x = x / 255.0

            h, w, _ = x.shape
            channel_sum += x.sum(axis=(0, 1))
            channel_sum_sq += (x * x).sum(axis=(0, 1))
            n_pixels += h * w

    if n_pixels == 0:
        raise RuntimeError(f"No pixels found under: {train_root_npy}")

    mean = channel_sum / n_pixels
    var = channel_sum_sq / n_pixels - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    eps = 1e-6
    std = np.maximum(std, eps)

    return mean.tolist(), std.tolist()


def main():
    project_root = Path(__file__).resolve().parents[1]  # .../main
    standardized_root = project_root / "src" / "fer" / "dataset" / "standardized"

    subset_root = standardized_root / "images_mtcnn_cropped_norm"
    train_root_npy = subset_root / "npy" / SPLIT

    if not train_root_npy.exists():
        raise RuntimeError(f"Missing directory: {train_root_npy}")

    mean, std = compute_mean_std_from_npy(train_root_npy)

    out_path = subset_root / "dataset_stats_train.json"
    payload = {
        "mean": mean,
        "std": std,
        "target_size": [64, 64],
        "split": SPLIT,
        "source": str(train_root_npy),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[saved] {out_path}")
    print(f"  mean: {mean}")
    print(f"  std : {std}")


if __name__ == "__main__":
    main()
