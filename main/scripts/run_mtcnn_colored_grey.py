from __future__ import annotations

from pathlib import Path
import shutil
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


def ensure_3ch_bgr(bgr_or_gray: np.ndarray) -> np.ndarray:
    """
    Ensure image is BGR uint8 with 3 channels.
    Accepts:
      - (H,W) uint8 gray
      - (H,W,3) uint8 BGR
      - (H,W,4) uint8 BGRA -> drops alpha
    """
    if bgr_or_gray.ndim == 2:
        return cv2.cvtColor(bgr_or_gray, cv2.COLOR_GRAY2BGR)
    if bgr_or_gray.ndim == 3 and bgr_or_gray.shape[2] == 3:
        return bgr_or_gray
    if bgr_or_gray.ndim == 3 and bgr_or_gray.shape[2] == 4:
        return bgr_or_gray[:, :, :3]
    raise ValueError(f"Unsupported shape: {bgr_or_gray.shape}")


def bgr_resize_to_uint8_3ch(bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    bgr = ensure_3ch_bgr(bgr)
    return cv2.resize(bgr, target_size, interpolation=cv2.INTER_LINEAR)


def bgr_to_gray3_uint8(bgr: np.ndarray) -> np.ndarray:
    bgr = ensure_3ch_bgr(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)          # (H,W)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)        # (H,W,3)
    return gray3


def bgr_uint8_to_npy_float01_hwc(bgr_u8: np.ndarray) -> np.ndarray:
    """
    uint8 (0..255) HWC -> float32 HWC in [0,1]
    """
    return (bgr_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def run_only_mtcnn_cropped(
    images_raw_root: Path,
    out_root_color_png: Path,
    out_root_color_npy: Path,
    out_root_grey_png: Path,
    out_root_grey_npy: Path,
    overwrite_outputs: bool = True,
    # --- MTCNN settings ---
    keep_all: bool = True,
    min_prob: float = 0.0,
    width_half: float = 1.3,
    crop_scale: float = 1.15,
    # --- processing ---
    target_size: tuple[int, int] = (64, 64),
):
    cropper = MTCNNFaceCropper(
        keep_all=keep_all,
        min_prob=min_prob,
        width_half=width_half,
        crop_scale=crop_scale,
    )

    # clean outputs
    if overwrite_outputs:
        for p in [out_root_color_png, out_root_color_npy, out_root_grey_png, out_root_grey_npy]:
            if p.exists():
                shutil.rmtree(p)

    for p in [out_root_color_png, out_root_color_npy, out_root_grey_png, out_root_grey_npy]:
        p.mkdir(parents=True, exist_ok=True)

    # process all splits/classes
    for split in SPLITS:
        for cls in CLASSES:
            in_dir = images_raw_root / split / cls

            out_dir_color_png = out_root_color_png / split / cls
            out_dir_color_npy = out_root_color_npy / split / cls
            out_dir_grey_png = out_root_grey_png / split / cls
            out_dir_grey_npy = out_root_grey_npy / split / cls

            out_dir_color_png.mkdir(parents=True, exist_ok=True)
            out_dir_color_npy.mkdir(parents=True, exist_ok=True)
            out_dir_grey_png.mkdir(parents=True, exist_ok=True)
            out_dir_grey_npy.mkdir(parents=True, exist_ok=True)

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
                    crop_bgr = pil_rgb_to_bgr_uint8(r.crop)  # guaranteed 3ch
                    crop_bgr = bgr_resize_to_uint8_3ch(crop_bgr, target_size)

                    # naming
                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    stem = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    # -------- color_and_grey (keep as is, but always 3ch) --------
                    out_png_color = out_dir_color_png / f"{stem}{OUT_EXT}"
                    out_npy_color = out_dir_color_npy / f"{stem}.npy"

                    cv2.imwrite(str(out_png_color), crop_bgr)  # uint8 BGR written as PNG
                    np.save(out_npy_color, bgr_uint8_to_npy_float01_hwc(crop_bgr))

                    # -------- grey (force grayscale->3ch) --------
                    crop_grey3 = bgr_to_gray3_uint8(crop_bgr)

                    out_png_grey = out_dir_grey_png / f"{stem}{OUT_EXT}"
                    out_npy_grey = out_dir_grey_npy / f"{stem}.npy"

                    cv2.imwrite(str(out_png_grey), crop_grey3)
                    np.save(out_npy_grey, bgr_uint8_to_npy_float01_hwc(crop_grey3))

    print("\nDone.")
    print(f"color_and_grey/png: {out_root_color_png}")
    print(f"color_and_grey/npy: {out_root_color_npy}")
    print(f"grey/png:          {out_root_grey_png}")
    print(f"grey/npy:          {out_root_grey_npy}")


def main():
    project_root = Path(__file__).resolve().parents[1]  # .../main
    dataset_root = project_root / "src" / "fer" / "dataset" / "standardized"

    images_raw_root = dataset_root / "images_raw"

    base_out = dataset_root / "only_mtcnn_cropped"

    out_root_color_png = base_out / "color_and_grey" / "png"
    out_root_color_npy = base_out / "color_and_grey" / "npy"

    out_root_grey_png = base_out / "grey" / "png"
    out_root_grey_npy = base_out / "grey" / "npy"

    run_only_mtcnn_cropped(
        images_raw_root=images_raw_root,
        out_root_color_png=out_root_color_png,
        out_root_color_npy=out_root_color_npy,
        out_root_grey_png=out_root_grey_png,
        out_root_grey_npy=out_root_grey_npy,
        overwrite_outputs=True,
        keep_all=True,
        min_prob=0.0,
        width_half=1.3,
        crop_scale=1.15,
        target_size=(64, 64),
    )


if __name__ == "__main__":
    main()
