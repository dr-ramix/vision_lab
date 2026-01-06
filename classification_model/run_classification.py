#   cd vision_lab
#   source venv/bin/activate
#
#   python classification_model/run_classification.py \
#     --images_dir classification_model_test_images/images/images/0 \
#     --model fer.models.cnn_resnet18:ResNet18FER \
#     --weights training_output/runs/2026-01-06_15-27-58__resnet18__user-bargozideh__e30877/exports/model_state_dict.pt \
#     --out_csv classification_model/classification_scores.csv

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --------------------------------------------------
# Repo paths / imports
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../vision_lab
SRC_ROOT = REPO_ROOT / "main" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper  # your MTCNN cropper

# --------------------------------------------------
# IMPORTANT: model output order MUST match training/dataloader mapping
# Your project mapping is:
#   0 anger, 1 disgust, 2 fear, 3 happiness, 4 sadness, 5 surprise
# --------------------------------------------------
TRAIN_CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# CSV order (as shown in your screenshot)
CSV_CLASS_ORDER = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]

# --------------------------------------------------
# Dataset stats (from your JSON)
# --------------------------------------------------
MEAN = np.array([0.5461214492863451, 0.5461214492863451, 0.5461214492863451], dtype=np.float32)
STD = np.array([0.22092840651221893, 0.22092840651221893, 0.22092840651221893], dtype=np.float32)
TARGET_SIZE = (64, 64)  # (W,H)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================
# Dynamic model loading (swappable)
# ============================
def _import_from_string(spec: str):
    """
    spec formats:
      - "pkg.module:ClassName"
      - "pkg.module.ClassName"
    """
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        mod, name = spec.rsplit(".", 1)

    module = __import__(mod, fromlist=[name])
    return getattr(module, name)


def build_model(model_spec: str, num_classes: int, model_kwargs: Dict[str, Any]) -> nn.Module:
    ModelCls = _import_from_string(model_spec)
    try:
        return ModelCls(num_classes=num_classes, **model_kwargs)
    except TypeError:
        return ModelCls(**model_kwargs)


def load_weights_if_available(
    model: nn.Module, weights_path: Optional[Path], device: torch.device
) -> Tuple[nn.Module, bool]:
    model.to(device)
    model.eval()

    if weights_path is None:
        print("[info] --weights not provided -> using random initialized weights")
        return model, False

    if not weights_path.exists():
        print(f"[warning] weights not found: {weights_path} -> using random initialized weights")
        return model, False

    print(f"[info] loading weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported weights format. Provide a state_dict or a dict with key 'state_dict'.")

    cleaned = {}
    for k, v in state.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, True


# ============================
# Preprocessing (EXACT training style)
# ============================
def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def crop_face_rgb_uint8(cropper: MTCNNFaceCropper, img_bgr: np.ndarray, face_index: int) -> Optional[np.ndarray]:
    """
    Uses your cropper.process_pil (which includes rotation normalization in your implementation)
    Returns: 64x64 RGB uint8 face crop, or None if no face.
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    results = cropper.process_pil(pil)
    if len(results) == 0 or face_index >= len(results):
        return None

    face_pil = results[face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
    return np.array(face_pil)  # RGB uint8


def preprocess_face_to_gray3_bgr01(face_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Training-style preprocessing:
      RGB -> BGR -> GRAYSCALE -> replicate to 3ch (BGR) -> [0,1]
    Returns: 64x64x3 float32 in [0,1] (gray replicated) in BGR order.
    """
    face_bgr = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2BGR)
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gray3 = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    return (face_gray3.astype(np.float32) / 255.0).clip(0.0, 1.0)


def normalize_gray3_bgr01_to_tensor(gray3_bgr_01: np.ndarray) -> torch.Tensor:
    """
    Input: HxWx3 float in [0,1], BGR with all channels equal (gray replicated).
    Output: 1x3xHxW torch tensor normalized by MEAN/STD.
    """
    chw = gray3_bgr_01.transpose(2, 0, 1).astype(np.float32, copy=False)  # 3xHxW
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw).unsqueeze(0)  # 1x3xHxW


def save_normalized_tensor_as_png(
    x_chw: np.ndarray,
    out_path: Path,
) -> bool:
    """
    x_chw: 3xHxW float (mean/std-normalized).
    Saves a *visualization* PNG by min-max scaling per-image to [0,255].
    """
    vis = x_chw.transpose(1, 2, 0)  # HWC
    vmin, vmax = float(vis.min()), float(vis.max())
    if vmax > vmin:
        vis = (vis - vmin) / (vmax - vmin)
    else:
        vis = np.zeros_like(vis, dtype=np.float32)
    vis_u8 = (vis * 255.0).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), vis_u8)
    return bool(ok)


# ============================
# File iteration
# ============================
def iter_images(images_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


# ============================
# Main
# ============================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str, help="Folder containing images")
    ap.add_argument(
        "--out_csv",
        default="classification_model/classification_scores.csv",
        type=str,
        help="CSV output path",
    )

    ap.add_argument("--model", required=True, type=str, help='Model spec, e.g. "fer.models.cnn_resnet18:ResNet18FER"')
    ap.add_argument("--model_kwargs_json", default="{}", type=str, help="JSON dict for model kwargs")

    ap.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Optional weights path (.pt). If missing/not found -> random weights.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--face_index", type=int, default=0, help="Which detected face to use (0 = first)")
    ap.add_argument(
        "--skip_no_face",
        action="store_true",
        help="If set, skip images where no face is found (otherwise write zeros)",
    )

    ap.add_argument(
        "--save_debug_faces",
        action="store_true",
        help="Save the actual 64x64 face crops used for inference (RGB crop after rotation+crop+resize)",
    )
    ap.add_argument("--debug_dir", default="classification_model/debug_faces", type=str, help="Where to save debug face crops")

    ap.add_argument(
        "--save_preprocessed",
        action="store_true",
        help="Save mean/std-normalized model inputs as PNG (visualized via min-max scaling)",
    )
    ap.add_argument(
        "--pre_dir",
        default="classification_model/images_pre_processed",
        type=str,
        help="Where to save preprocessed normalized PNGs",
    )

    args = ap.parse_args()

    images_dir = (REPO_ROOT / args.images_dir).resolve() if not os.path.isabs(args.images_dir) else Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    out_csv = (REPO_ROOT / args.out_csv).resolve() if not os.path.isabs(args.out_csv) else Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    weights_path = None
    if args.weights is not None:
        weights_path = (REPO_ROOT / args.weights).resolve() if not os.path.isabs(args.weights) else Path(args.weights)

    debug_dir = (REPO_ROOT / args.debug_dir).resolve() if not os.path.isabs(args.debug_dir) else Path(args.debug_dir)
    pre_dir = (REPO_ROOT / args.pre_dir).resolve() if not os.path.isabs(args.pre_dir) else Path(args.pre_dir)

    if args.save_debug_faces:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preprocessed:
        pre_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] preprocessed images dir -> {pre_dir}")

    device = torch.device(args.device)

    model_kwargs = json.loads(args.model_kwargs_json)
    if not isinstance(model_kwargs, dict):
        raise ValueError("--model_kwargs_json must be a JSON object/dict")

    # Safety: CSV labels must be a permutation of training labels
    if set(CSV_CLASS_ORDER) != set(TRAIN_CLASS_ORDER):
        raise ValueError(
            "CSV_CLASS_ORDER and TRAIN_CLASS_ORDER must contain the same labels.\n"
            f"TRAIN: {TRAIN_CLASS_ORDER}\nCSV:   {CSV_CLASS_ORDER}"
        )

    cropper = MTCNNFaceCropper(keep_all=True, min_prob=0.0)

    model = build_model(args.model, num_classes=len(TRAIN_CLASS_ORDER), model_kwargs=model_kwargs)
    model, loaded = load_weights_if_available(model, weights_path, device)

    img_paths = iter_images(images_dir, recursive=args.recursive)
    if len(img_paths) == 0:
        print(f"[warning] no images found in: {images_dir}")
        return 0

    print(f"[info] found {len(img_paths)} images")
    print(f"[info] writing csv -> {out_csv}")
    print(f"[info] weights={'loaded' if loaded else 'random'}")
    print("[info] preprocessing: mtcnn->(rot)->crop->resize64 -> gray -> gray3 -> normalize(mean/std)")

    train_index = {name: i for i, name in enumerate(TRAIN_CLASS_ORDER)}
    header = ["filepath"] + CSV_CLASS_ORDER

    saved_pre = 0
    saved_dbg = 0
    no_face = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for p in img_paths:
            img_bgr = load_image_bgr(p)
            if img_bgr is None:
                print(f"[skip] unreadable image: {p}")
                continue

            # NOTE: rotation normalization happens inside cropper.process_pil(...)
            face_rgb = crop_face_rgb_uint8(cropper, img_bgr, face_index=args.face_index)
            if face_rgb is None:
                no_face += 1
                if args.skip_no_face:
                    print(f"[skip] no face: {p}")
                    continue
                w.writerow([str(p)] + [0.0] * len(CSV_CLASS_ORDER))
                continue

            if args.save_debug_faces:
                out_dbg = debug_dir / f"{p.stem}_face{args.face_index}.png"
                ok = cv2.imwrite(str(out_dbg), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                if ok:
                    saved_dbg += 1
                else:
                    print(f"[warning] failed to write debug face: {out_dbg}")

            gray3_bgr01 = preprocess_face_to_gray3_bgr01(face_rgb)  # 64x64x3 float [0,1] BGR (gray replicated)

            # model input tensor (normalized)
            x = normalize_gray3_bgr01_to_tensor(gray3_bgr01).to(device)

            if args.save_preprocessed:
                # Save visualization of the exact normalized input (CHW)
                x_chw = x[0].detach().cpu().numpy()  # 3xHxW
                out_pre = pre_dir / f"{p.stem}_face{args.face_index}_norm.png"
                ok = save_normalized_tensor_as_png(x_chw, out_pre)
                if ok:
                    saved_pre += 1
                else:
                    print(f"[warning] failed to write preprocessed image: {out_pre}")

            with torch.no_grad():
                logits = model(x)  # 1xC
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

            scores = [float(probs[train_index[label]]) for label in CSV_CLASS_ORDER]
            w.writerow([str(p)] + scores)

    if args.save_debug_faces:
        print(f"[info] saved debug faces: {saved_dbg} -> {debug_dir}")
    if args.save_preprocessed:
        print(f"[info] saved preprocessed images: {saved_pre} -> {pre_dir}")
    print(f"[info] images with no face: {no_face}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
