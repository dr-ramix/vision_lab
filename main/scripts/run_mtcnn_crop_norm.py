# main/scripts/run_mtcnn_crop_norm.py
# Läuft über:   src/fer/dataset/standardized/images_raw/{train,val,test}/{class}/*
#
# Schreibt nach:
#   src/fer/dataset/standardized/images_mtcnn_cropped_norm/png/{train,val,test}/{class}/*.png
#   src/fer/dataset/standardized/images_mtcnn_cropped_norm/npy/{train,val,test}/{class}/*.npy
#
# Überschreibt IMMER vorhandene Outputs (pro Run, pro Split/Class).
#
# Pipeline:
#   1) MTCNN Face Crop (PIL RGB)
#   2) Crop -> cv2 BGR uint8
#   3) FORCE GREY: BGR -> Gray(1ch)
#   4) CLAHE auf Gray (uint8)
#   5) Stack Gray -> 3 Kanäle (identische Kanäle)
#   6) Save outputs:
#        PNG: uint8 (64,64,3)
#        NPY: float32 (64,64,3) in [0,1]

from __future__ import annotations

from pathlib import Path
import shutil
import cv2
import numpy as np

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper

SPLITS = ["train", "val", "test"]
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EXTS = (".jpg", ".jpeg", ".png")

OUT_PNG_EXT = ".png"
OUT_NPY_EXT = ".npy"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def pil_rgb_to_bgr_uint8(pil_img) -> np.ndarray:
    """PIL RGB -> cv2 BGR uint8"""
    rgb = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_gray_u8(bgr_u8: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) -> GRAY uint8 (H,W)"""
    if bgr_u8.ndim != 3 or bgr_u8.shape[2] != 3:
        raise ValueError(f"Expected BGR (H,W,3), got {bgr_u8.shape}")
    if bgr_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {bgr_u8.dtype}")
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)


def gray_u8_to_3ch_bgr_u8(gray_u8: np.ndarray) -> np.ndarray:
    """GRAY uint8 (H,W) -> BGR uint8 (H,W,3) with identical channels"""
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def bgr_u8_to_f32_01(bgr_u8: np.ndarray) -> np.ndarray:
    """BGR uint8 [0,255] -> float32 [0,1]"""
    return (bgr_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def run_preprocessing(
    images_raw_root: Path,
    out_root: Path,
    overwrite_outputs: bool = True,
    # --- MTCNN Einstellungen ---
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    # --- CLAHE Einstellungen ---
    target_size: tuple[int, int] = (64, 64),
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
):
    cropper = MTCNNFaceCropper(
        keep_all=keep_all,
        min_prob=min_prob,
        width_half=width_half,
    )

    # CLAHE object (OpenCV)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))

    out_root.mkdir(parents=True, exist_ok=True)

    # separate subfolders for png + npy
    out_png_root = out_root / "png"
    out_npy_root = out_root / "npy"
    out_png_root.mkdir(parents=True, exist_ok=True)
    out_npy_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = images_raw_root / split / cls

            out_png_dir = out_png_root / split / cls
            out_npy_dir = out_npy_root / split / cls

            if overwrite_outputs:
                if out_png_dir.exists():
                    shutil.rmtree(out_png_dir)
                if out_npy_dir.exists():
                    shutil.rmtree(out_npy_dir)

            out_png_dir.mkdir(parents=True, exist_ok=True)
            out_npy_dir.mkdir(parents=True, exist_ok=True)

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
                    # 1) MTCNN crop returns PIL RGB
                    crop_bgr = pil_rgb_to_bgr_uint8(r.crop)

                    # 2) resize to target_size (64x64) BEFORE CLAHE (consistent output)
                    crop_bgr = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_LINEAR)

                    # 3) force grayscale
                    gray = to_gray_u8(crop_bgr)

                    # 4) CLAHE on grayscale
                    gray_clahe = clahe.apply(gray)

                    # 5) stack to 3 channels (BGR)
                    out_bgr_u8 = gray_u8_to_3ch_bgr_u8(gray_clahe)

                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    base = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    out_png_path = out_png_dir / f"{base}{OUT_PNG_EXT}"
                    out_npy_path = out_npy_dir / f"{base}{OUT_NPY_EXT}"

                    # PNG: uint8 (64,64,3)
                    cv2.imwrite(str(out_png_path), out_bgr_u8)

                    # NPY: float32 (64,64,3) in [0,1]
                    out_f32 = bgr_u8_to_f32_01(out_bgr_u8)
                    np.save(str(out_npy_path), out_f32)

    print(f"\nDone.")
    print(f"  PNG root: {out_png_root}")
    print(f"  NPY root: {out_npy_root}")


def main():
    project_root = Path(__file__).resolve().parents[1]  # .../main
    dataset_root = project_root / "src" / "fer" / "dataset" / "standardized"

    images_raw_root = dataset_root / "images_raw"
    out_root = dataset_root / "images_mtcnn_cropped_norm"

    run_preprocessing(
        images_raw_root=images_raw_root,
        out_root=out_root,
        overwrite_outputs=True,
        keep_all=True,
        min_prob=0.0,
        width_half=1.3,
        target_size=(64, 64),
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )


if __name__ == "__main__":
    main()
