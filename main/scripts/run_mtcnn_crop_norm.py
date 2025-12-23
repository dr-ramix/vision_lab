# main/scripts/run_mtcnn_crop_norm.py
# Läuft über:   src/fer/dataset/standardized/images_raw/{train,val,test}/{class}/*
# Schreibt nach: src/fer/dataset/standardized/images_mtcnn_cropped_norm/{train,val,test}/{class}/*
#
# Überschreibt IMMER vorhandene Outputs (pro Run).
#
# Requirements:
#   pip install facenet-pytorch pillow torch torchvision opencv-python numpy
#
# WICHTIG:
# - Dieses Script nutzt deine Klassen:
#     - MTCNNFaceCropper (Rotation+Paper-Crop)
#     - BasicImageProcessor (Resize 64x64 + intensity_norm_paper_look)
#
# Passe ggf. die Import-Pfade unten an deine Projektstruktur an.

from __future__ import annotations

from pathlib import Path
import shutil
import cv2
import numpy as np

from main.src.fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper
from main.src.fer.pre_processing.basic_img_norms import BasicImageProcessor

SPLITS = ["train", "val", "test"]
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EXTS = (".jpg", ".jpeg", ".png")

# Output-Dateiformat (PNG empfohlen, weil norm-Bilder sonst JPG-Artefakte bekommen)
OUT_EXT = ".png"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def pil_rgb_to_bgr_uint8(pil_img) -> np.ndarray:
    """PIL RGB -> cv2 BGR uint8"""
    rgb = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
    """
    Iteriert über splits/classes in images_raw_root und erstellt eine parallele Struktur in out_root.
    Für jedes Bild:
      - MTCNNFaceCropper -> 1..N Face crops (je nach keep_all)
      - BasicImageProcessor -> Resize + intensity_norm (wie dein Code)
      - Speichern in out_root/{split}/{class}/...
    """

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

            if overwrite_outputs:
                if out_dir.exists():
                    shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"[skip] {split}/{cls}: input dir missing -> {in_dir}")
                continue

            img_paths = [p for p in in_dir.iterdir() if is_image_file(p)]
            print(f"\n=== {split}/{cls} | images: {len(img_paths)} ===")

            for img_path in img_paths:
                # 1) MTCNN crop(s)
                results = cropper.process_path(img_path)
                if not results:
                    # kein Gesicht gefunden
                    continue

                # Für jedes erkannte Gesicht speichern (face0, face1, ...)
                for r in results:
                    # 2) Basic processing auf Crop
                    crop_bgr = pil_rgb_to_bgr_uint8(r.crop)
                    proc = basic.process_bgr(crop_bgr)

                    # Output filename: <orig>_face{i}_pX.png
                    prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
                    out_path = out_dir / f"{img_path.stem}_face{r.face_index}_{prob_str}{OUT_EXT}"

                    # Überschreiben: cv2.imwrite überschreibt automatisch
                    cv2.imwrite(str(out_path), proc.normalized_gray_vis)

    print(f"\nDone. Output root: {out_root}")


def main():
    # Projektpfade (Script liegt in main/scripts/)
    project_root = Path(__file__).resolve().parents[1]  # .../main
    dataset_root = project_root / "src" / "fer" / "dataset" / "standardized"

    images_raw_root = dataset_root / "images_raw"
    out_root = dataset_root / "images_mtcnn_cropped_norm"

    run_preprocessing(
        images_raw_root=images_raw_root,
        out_root=out_root,
        overwrite_outputs=True,  # <- löscht pro split/class den Output-Ordner und schreibt neu
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
