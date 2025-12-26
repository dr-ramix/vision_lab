# main/scripts/run_mtcnn_crop_norm.py
# Läuft über:   src/fer/dataset/standardized/images_raw/{train,val,test}/{class}/*
# Schreibt nach: src/fer/dataset/standardized/images_mtcnn_cropped_norm/{train,val,test}/{class}/*
#
# Überschreibt IMMER vorhandene Outputs (pro Run).
#
# Output ist jetzt 3-kanalig (normiertes Grau auf 3 Kanäle gestackt).

from __future__ import annotations

from pathlib import Path
import shutil
import cv2
import numpy as np

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper
from fer.pre_processing.basic_img_norms import BasicImageProcessor

SPLITS = ["train", "val", "test"]
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EXTS = (".jpg", ".jpeg", ".png")
OUT_EXT = ".png"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def pil_rgb_to_bgr_uint8(pil_img) -> np.ndarray:
    """PIL RGB -> cv2 BGR uint8"""
    rgb = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def gray_to_3ch(gray_u8: np.ndarray) -> np.ndarray:
    """(H,W) uint8 -> (H,W,3) uint8"""
    # gleichwertig zu np.stack([gray,gray,gray], axis=-1)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def run_preprocessing(
    images_raw_root: Path,
    out_root: Path,
    overwrite_outputs: bool = True,
    # --- MTCNN Einstellungen ---
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    # --- Basic Processing Einstellungen ---
    target_size=(64, 64),
    ksize: int = 7,
    sigma_floor: float = 10.0,
    post_blur_ksize: int = 3,
    tanh_scale: float = 2.5,
):
    cropper = MTCNNFaceCropper(
        keep_all=keep_all,
        min_prob=min_prob,
        width_half=width_half,
    )

    basic = BasicImageProcessor(
        target_size=target_size,
        ksize=ksize,
        sigma_floor=sigma_floor,
        post_blur_ksize=post_blur_ksize,
        tanh_scale=tanh_scale,
    )

    out_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = images_raw_root / split / cls
            out_dir = out_root / split / cls

            if overwrite_outputs and out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

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
                    proc = basic.process_bgr(crop_bgr)

                    # normiertes Grau auf 3 Kanäle stacken
                    norm_bgr_3ch = gray_to_3ch(proc.normalized_gray_vis)

                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    out_path = out_dir / f"{img_path.stem}_face{r.face_index}_{prob_str}{OUT_EXT}"

                    cv2.imwrite(str(out_path), norm_bgr_3ch)

    print(f"\nDone. Output root: {out_root}")


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
        ksize=7,
        sigma_floor=10.0,
        post_blur_ksize=3,
        tanh_scale=2.5,
    )


if __name__ == "__main__":
    main()
