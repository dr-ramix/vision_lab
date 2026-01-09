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


# ============================================================
# We will STORE .npy in RGB order (HWC), float32 [0,1].
# For PNG writing via OpenCV, convert RGB->BGR ONLY at imwrite time.
# ============================================================

def pil_rgb_to_rgb_uint8(pil_img) -> np.ndarray:
    """
    Convert PIL RGB image to numpy RGB uint8 (H,W,3).
    """
    rgb = np.array(pil_img, dtype=np.uint8)

    # be robust to grayscale or RGBA outputs
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 (H,W,3), got shape={rgb.shape}")

    return rgb


def ensure_3ch_rgb(rgb_or_gray: np.ndarray) -> np.ndarray:
    """
    Ensure image is RGB uint8 with 3 channels.
    Accepts:
      - (H,W) uint8 gray
      - (H,W,3) uint8 RGB
      - (H,W,4) uint8 RGBA -> drops alpha
    """
    if rgb_or_gray.ndim == 2:
        return cv2.cvtColor(rgb_or_gray, cv2.COLOR_GRAY2RGB)
    if rgb_or_gray.ndim == 3 and rgb_or_gray.shape[2] == 3:
        return rgb_or_gray
    if rgb_or_gray.ndim == 3 and rgb_or_gray.shape[2] == 4:
        return rgb_or_gray[:, :, :3]
    raise ValueError(f"Unsupported shape: {rgb_or_gray.shape}")


def rgb_resize_to_uint8_3ch(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    rgb = ensure_3ch_rgb(rgb)
    return cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)


def rgb_to_gray3_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    RGB -> gray -> 3-channel RGB (H,W,3) so downstream stays 3ch everywhere.
    """
    rgb = ensure_3ch_rgb(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)          # (H,W)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)        # (H,W,3)
    return gray3


def rgb_uint8_to_npy_float01_hwc(rgb_u8: np.ndarray) -> np.ndarray:
    """
    uint8 (0..255) RGB HWC -> float32 RGB HWC in [0,1]
    """
    rgb_u8 = ensure_3ch_rgb(rgb_u8)
    return (rgb_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def write_png_rgb_with_cv2(out_path: Path, rgb_u8: np.ndarray) -> None:
    """
    OpenCV expects BGR. Convert RGB->BGR only for writing PNG.
    """
    rgb_u8 = ensure_3ch_rgb(rgb_u8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


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
                    # r.crop is PIL RGB -> convert to numpy RGB
                    crop_rgb = pil_rgb_to_rgb_uint8(r.crop)  # guaranteed 3ch RGB
                    crop_rgb = rgb_resize_to_uint8_3ch(crop_rgb, target_size)

                    # naming
                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    stem = f"{img_path.stem}_face{r.face_index}_{prob_str}"

                    # -------- color_and_grey (store RGB) --------
                    out_png_color = out_dir_color_png / f"{stem}{OUT_EXT}"
                    out_npy_color = out_dir_color_npy / f"{stem}.npy"

                    # PNG: write with OpenCV (needs BGR), so convert for write only
                    write_png_rgb_with_cv2(out_png_color, crop_rgb)
                    # NPY: store RGB float [0,1]
                    np.save(out_npy_color, rgb_uint8_to_npy_float01_hwc(crop_rgb))

                    # -------- grey (force grayscale->3ch RGB) --------
                    crop_grey3 = rgb_to_gray3_uint8(crop_rgb)

                    out_png_grey = out_dir_grey_png / f"{stem}{OUT_EXT}"
                    out_npy_grey = out_dir_grey_npy / f"{stem}.npy"

                    write_png_rgb_with_cv2(out_png_grey, crop_grey3)
                    np.save(out_npy_grey, rgb_uint8_to_npy_float01_hwc(crop_grey3))

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
