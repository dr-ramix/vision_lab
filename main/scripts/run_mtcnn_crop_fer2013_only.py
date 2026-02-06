from __future__ import annotations

from pathlib import Path
import shutil
import cv2
import numpy as np

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper

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


def to_float01_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> RGB float32 in [0,1]"""
    return (rgb_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def resize_bgr_uint8(bgr_u8: np.ndarray, target_size=(64, 64)) -> np.ndarray:
    """(H,W,3) BGR uint8 -> resized to target_size (W,H) in cv2 convention"""
    tw, th = int(target_size[0]), int(target_size[1])
    return cv2.resize(bgr_u8, (tw, th), interpolation=cv2.INTER_AREA)


def run_preprocessing(
    images_raw_root: Path,
    out_root: Path,
    overwrite_outputs: bool = True,
    # --- MTCNN Einstellungen ---
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    # --- Resize Einstellungen 
    target_size=(64, 64),
):
    cropper = MTCNNFaceCropper(
        keep_all=keep_all,
        min_prob=min_prob,
        width_half=width_half,
    )

    # out_root/.../fer2013_mtcnn_cropped
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
                    crop_bgr = pil_rgb_to_bgr_uint8(r.crop)  # PIL RGB -> BGR u8

                    # 64x64 
                    crop_bgr = resize_bgr_uint8(crop_bgr, target_size=target_size)

                    # PNG speichern: BGR u8
                    png_bgr_u8 = crop_bgr

                    # NPY speichern: float32 [0,1] in RGB
                    crop_rgb_u8 = bgr_to_rgb_uint8(crop_bgr)
                    npy_rgb_float = to_float01_rgb(crop_rgb_u8)

                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    base = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    out_path_png = out_dir_png / f"{base}{PNG_EXT}"
                    out_path_npy = out_dir_npy / f"{base}{NPY_EXT}"

                    cv2.imwrite(str(out_path_png), png_bgr_u8)
                    np.save(str(out_path_npy), npy_rgb_float)

    print(f"\nDone.")
    print(f"PNG root: {png_root}")
    print(f"NPY root: {npy_root}")


def main():
    project_root = Path(__file__).resolve().parents[1]  # .../main
    dataset_root = project_root / "src" / "fer" / "dataset" / "standardized" / "fer2013"

    images_raw_root = dataset_root / "fer2013_raw"
    out_root = dataset_root / "fer2013_mtcnn_cropped"  

    run_preprocessing(
        images_raw_root=images_raw_root,
        out_root=out_root,
        overwrite_outputs=True,
        keep_all=True,
        min_prob=0.0,
        width_half=1.3,
        target_size=(64, 64),
    )


if __name__ == "__main__":
    main()
