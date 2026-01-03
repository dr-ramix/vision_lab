from __future__ import annotations

from pathlib import Path
import shutil
import json
import cv2
import numpy as np

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper

SPLITS = ["train", "val", "test"]
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EXTS = (".jpg", ".jpeg", ".png")
OUT_EXT = ".png"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def pil_rgb_to_bgr_uint8(pil_img) -> np.ndarray:
    rgb = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def gray_to_3ch(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def bgr_to_gray3_resized(
    bgr: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      resized_bgr: (H,W,3) uint8
      gray_u8:     (H,W) uint8
      gray3_u8:    (H,W,3) uint8 (stacked)
    """
    resized = cv2.resize(bgr, target_size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray3 = gray_to_3ch(gray)
    return resized, gray, gray3


def compute_train_mean_std(
    images_raw_root: Path,
    cropper: MTCNNFaceCropper,
    target_size: tuple[int, int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Mean/Std auf TRAIN berechnen (nach MTCNN crop + resize + gray->3ch).
    Werte sind in [0,1]-Skala.
    """
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    split = "train"
    for cls in CLASSES:
        in_dir = images_raw_root / split / cls
        if not in_dir.exists():
            continue

        img_paths = [p for p in in_dir.iterdir() if is_image_file(p)]
        print(f"[mean/std] {split}/{cls} | images: {len(img_paths)}")

        for img_path in img_paths:
            results = cropper.process_path(img_path)
            if not results:
                continue

            for r in results:
                crop_bgr = pil_rgb_to_bgr_uint8(r.crop)
                _, _, gray3 = bgr_to_gray3_resized(crop_bgr, target_size)

                x = gray3.astype(np.float64) / 255.0  # (H,W,3)
                h, w, _ = x.shape

                channel_sum += x.sum(axis=(0, 1))
                channel_sum_sq += (x ** 2).sum(axis=(0, 1))
                n_pixels += h * w

    mean = channel_sum / max(n_pixels, 1)
    std = np.sqrt(channel_sum_sq / max(n_pixels, 1) - mean ** 2)

    return tuple(mean.tolist()), tuple(std.tolist())


def normalize_gray3(
    gray3_u8: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    to_chw: bool = True,
) -> np.ndarray:
    """
    uint8 (0..255) -> float32 normalized with dataset mean/std (train-derived).
    Output: float32, CHW oder HWC.
    """
    x = gray3_u8.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr

    if to_chw:
        x = np.transpose(x, (2, 0, 1))  # (3,H,W)
    return x.astype(np.float32, copy=False)


def run_preprocessing(
    images_raw_root: Path,
    out_root_png: Path,
    out_root_npy: Path,
    overwrite_outputs: bool = True,
    # --- MTCNN Einstellungen ---
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    # --- Basic Processing Einstellungen ---
    target_size: tuple[int, int] = (64, 64),
    # --- Output ---
    save_png_uint8: bool = True,
    save_npy_normalized: bool = True,
    to_chw: bool = True,
):
    cropper = MTCNNFaceCropper(
        keep_all=keep_all,
        min_prob=min_prob,
        width_half=width_half,
    )

    # 1) Mean/Std nur auf TRAIN berechnen
    mean, std = compute_train_mean_std(
        images_raw_root=images_raw_root,
        cropper=cropper,
        target_size=target_size,
    )
    print(f"\nComputed train mean/std:\n  mean={mean}\n  std ={std}\n")

    # speichern (damit es reproduzierbar ist)
    stats_path = out_root_npy.parent / "dataset_stats_train.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std, "target_size": list(target_size)}, f, indent=2)
    print(f"[saved] {stats_path}")

    # 2) Outputs vorbereiten
    if overwrite_outputs:
        if out_root_png.exists():
            shutil.rmtree(out_root_png)
        if out_root_npy.exists():
            shutil.rmtree(out_root_npy)

    out_root_png.mkdir(parents=True, exist_ok=True)
    out_root_npy.mkdir(parents=True, exist_ok=True)

    # 3) Alle Splits identisch preprocessen (und Normalisierung mit TRAIN-stats anwenden!)
    for split in SPLITS:
        for cls in CLASSES:
            in_dir = images_raw_root / split / cls
            out_dir_png = out_root_png / split / cls
            out_dir_npy = out_root_npy / split / cls

            out_dir_png.mkdir(parents=True, exist_ok=True)
            out_dir_npy.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"[skip] {split}/{cls}: input dir missing -> {in_dir}")
                continue

            img_paths = [p for p in in_dir.iterdir() if is_image_file(p)]
            print(f"\n=== {split}/{cls} | images: {len(img_paths)} ===")

            for img_path in img_paths:
                results = cropper.process_path(img_path)
                if not results:
                    continue

                for r in results:
                    crop_bgr = pil_rgb_to_bgr_uint8(r.crop)
                    _, _, gray3 = bgr_to_gray3_resized(crop_bgr, target_size)

                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    stem = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    # A) PNG (uint8) speichern: grayscale->3ch, ohne mean/std
                    if save_png_uint8:
                        out_path_png = out_dir_png / f"{stem}{OUT_EXT}"
                        cv2.imwrite(str(out_path_png), gray3)

                    # B) NPY (float32) speichern: mean/std-normalisiert (train-stats)
                    if save_npy_normalized:
                        x_norm = normalize_gray3(gray3, mean=mean, std=std, to_chw=to_chw)
                        out_path_npy = out_dir_npy / f"{stem}.npy"
                        np.save(out_path_npy, x_norm)

    print(f"\nDone.")
    print(f"PNG output root: {out_root_png}")
    print(f"NPY output root: {out_root_npy}")


def main():
    project_root = Path(__file__).resolve().parents[1]  # .../main
    dataset_root = project_root / "src" / "fer" / "dataset" / "standardized" 

    images_raw_root = dataset_root / "images_raw"
    out_root_png = dataset_root / "images_mtcnn_cropped" / "png"         # uint8 png
    out_root_npy = dataset_root / "images_mtcnn_cropped"/ "npy"    # float32 npy

    run_preprocessing(
        images_raw_root=images_raw_root,
        out_root_png=out_root_png,
        out_root_npy=out_root_npy,
        overwrite_outputs=True,
        keep_all=True,
        min_prob=0.0,
        width_half=1.3,
        target_size=(64, 64),
        save_png_uint8=True,
        save_npy_normalized=True,
        to_chw=True,   # für PyTorch üblich; falls euer Code HWC erwartet -> False
    )


if __name__ == "__main__":
    main()
