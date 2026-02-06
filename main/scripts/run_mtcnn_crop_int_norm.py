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
PNG_EXT = ".png"
NPY_EXT = ".npy"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def pil_rgb_to_bgr_uint8(pil_img) -> np.ndarray:
    """PIL RGB -> cv2 BGR uint8"""
    rgb = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_uint8(bgr_u8: np.ndarray) -> np.ndarray:
    """cv2 BGR uint8 -> RGB uint8"""
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)


def gray_to_3ch_bgr_u8(gray_u8: np.ndarray) -> np.ndarray:
    """(H,W) uint8 -> (H,W,3) uint8 in BGR (cv2 convention)"""
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def to_float01_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> RGB float32 in [0,1]"""
    return (rgb_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def run_preprocessing(
    images_raw_root: Path,
    out_root: Path,
    overwrite_outputs: bool = True,
    # MTCNN settings 
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    # Basic processing settings
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

    # Target root
    out_root.mkdir(parents=True, exist_ok=True)

    png_root = out_root / "png"
    npy_root = out_root / "npy"

    if overwrite_outputs:
        if png_root.exists():
            shutil.rmtree(png_root)
        if npy_root.exists():
            shutil.rmtree(npy_root)

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = images_raw_root / split / cls
            out_dir_png = png_root / split / cls
            out_dir_npy = npy_root / split / cls

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
                    proc = basic.process_bgr(crop_bgr)

                    # PNG
                    png_bgr_3ch = gray_to_3ch_bgr_u8(proc.normalized_gray_vis)

                    # NPY
                    png_rgb_3ch_u8 = bgr_to_rgb_uint8(png_bgr_3ch)
                    npy_rgb_float = to_float01_rgb(png_rgb_3ch_u8)  # (H,W,3), float32 in [0,1]

                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    base = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    out_path_png = out_dir_png / f"{base}{PNG_EXT}"
                    out_path_npy = out_dir_npy / f"{base}{NPY_EXT}"

                    cv2.imwrite(str(out_path_png), png_bgr_3ch)
                    np.save(str(out_path_npy), npy_rgb_float)

    print("\nDone.")
    print(f"PNG root: {png_root}")
    print(f"NPY root: {npy_root}")


def main():
    project_root = Path(__file__).resolve().parents[1]

    images_raw_root = (
        project_root / "src" / "fer" / "dataset" / "standardized" / "images_raw"
    )

    out_root = (
        project_root
        / "src"
        / "fer"
        / "dataset"
        / "standardized"
        / "mtcnn_cropped_int_norm"
    )

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
