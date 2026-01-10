#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import MTCNN


# BASE DIR (script folder): model + weights here too
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# DEFAULT PATHS (EDITABLE)
DEFAULT_IMAGES_DIR = BASE_DIR / "images"          # <-- HIER: Default input folder
DEFAULT_OUT_DIR = BASE_DIR / "output"             # <-- HIER: Default output folder for CSV
DEFAULT_OUT_NAME = "classification_scores.csv"    # <-- HIER: Default CSV file name
DEFAULT_NPY_DIR = BASE_DIR / "npy_preprocessed"   # <-- Optional: only used if --save_npy is set


# Class order
# training order: 0 anger, 1 disgust, 2 fear, 3 happiness, 4 sadness, 5 surprise
TRAIN_CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CSV_CLASS_ORDER = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]


# Train stats (computed from npy in [0,1]) 
MEAN = np.array([0.5426446906981507, 0.5426446906981507, 0.5426446906981507], dtype=np.float32)
STD = np.array([0.22369591629278052, 0.22369591629278052, 0.22369591629278052], dtype=np.float32)
TARGET_SIZE = (64, 64)  # (W,H)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



# MTCNN face cropper (inlined)
@dataclass
class FaceCropResult:
    face_index: int
    prob: Optional[float]
    crop: Image.Image
    eye_angle_before: float
    residual_angle_after: float
    used_rotation_sign: str  # "A(-eye_angle)" or "B(+eye_angle)"


class MTCNNFaceCropper:
 
    def __init__(
        self,
        keep_all: bool = True,
        min_prob: float = 0.0,
        width_half: float = 1.3,
        device: Optional[str] = None,
        crop_scale: float = 1.15,
    ):
        self.keep_all = keep_all
        self.min_prob = float(min_prob)
        self.width_half = float(width_half)
        self.crop_scale = float(crop_scale)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.mtcnn = MTCNN(keep_all=self.keep_all, device=self.device)

    @staticmethod
    def _rotate_image_and_points_cv2(
        pil_img: Image.Image,
        points: List[Tuple[float, float]],
        angle_deg: float,
        center_xy: Tuple[float, float],
    ) -> Tuple[Image.Image, List[Tuple[float, float]]]:
        img = np.array(pil_img)  # RGB
        h, w = img.shape[:2]
        cx, cy = center_xy

        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        rot = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        pts = np.array(points, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
        pts_rot = (M @ pts_h.T).T

        rot_pil = Image.fromarray(rot)
        return rot_pil, [tuple(p) for p in pts_rot]

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    @staticmethod
    def _angle_from_pts(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    def process_pil(self, img: Image.Image) -> List[FaceCropResult]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        boxes, probs, lms = self.mtcnn.detect(img, landmarks=True)
        if boxes is None or lms is None:
            return []

        if probs is None:
            probs = [1.0] * len(lms)

        results: List[FaceCropResult] = []

        for i, (prob, lm) in enumerate(zip(probs, lms)):
            if prob is not None and float(prob) < self.min_prob:
                continue

            left_eye = tuple(lm[0])
            right_eye = tuple(lm[1])

            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            eye_angle = math.degrees(math.atan2(dy, dx))

            center = (
                (left_eye[0] + right_eye[0]) / 2.0,
                (left_eye[1] + right_eye[1]) / 2.0,
            )

            # Option A: -eye_angle
            rot_img_a, (l_a, r_a) = self._rotate_image_and_points_cv2(
                img, [left_eye, right_eye], angle_deg=-eye_angle, center_xy=center
            )
            res_a = abs(self._angle_from_pts(l_a, r_a))

            # Option B: +eye_angle
            rot_img_b, (l_b, r_b) = self._rotate_image_and_points_cv2(
                img, [left_eye, right_eye], angle_deg=+eye_angle, center_xy=center
            )
            res_b = abs(self._angle_from_pts(l_b, r_b))

            if res_a <= res_b:
                rot_img, rot_left, rot_right = rot_img_a, l_a, r_a
                used = "A(-eye_angle)"
                residual = res_a
            else:
                rot_img, rot_left, rot_right = rot_img_b, l_b, r_b
                used = "B(+eye_angle)"
                residual = res_b

            # Paper crop after rotation
            mx = (rot_left[0] + rot_right[0]) / 2.0
            my = (rot_left[1] + rot_right[1]) / 2.0
            alpha = math.hypot(rot_right[0] - mx, rot_right[1] - my)
            alpha = alpha * self.crop_scale

            x1 = mx - self.width_half * alpha
            x2 = mx + self.width_half * alpha
            y1 = my - 1.3 * alpha
            y2 = my + 3.2 * alpha

            W, H = rot_img.size
            x1 = self._clamp(x1, 0, W)
            x2 = self._clamp(x2, 0, W)
            y1 = self._clamp(y1, 0, H)
            y2 = self._clamp(y2, 0, H)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = rot_img.crop((x1, y1, x2, y2))

            results.append(
                FaceCropResult(
                    face_index=i,
                    prob=None if prob is None else float(prob),
                    crop=crop,
                    eye_angle_before=float(eye_angle),
                    residual_angle_after=float(residual),
                    used_rotation_sign=used,
                )
            )

        return results



# Model loading (from same folder as script)
def _import_from_string(spec: str):
    """
    spec:
      - "module:ClassName"
      - "module.ClassName"
    """
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        mod, name = spec.rsplit(".", 1)
    module = __import__(mod, fromlist=[name])
    return getattr(module, name)


def build_model(model_spec: str, num_classes: int) -> nn.Module:
    ModelCls = _import_from_string(model_spec)
    try:
        return ModelCls(num_classes=num_classes)
    except TypeError:
        return ModelCls()


def _pick_default_weights_path() -> Optional[Path]:
    for name in ["model_state_dict.pt", "model_state_dict.pth", "weights.pt", "weights.pth", "best.pt", "best.pth"]:
        p = BASE_DIR / name
        if p.exists():
            return p
    cands = sorted(list(BASE_DIR.glob("*.pt")) + list(BASE_DIR.glob("*.pth")))
    return cands[0] if cands else None


def load_weights(model: nn.Module, weights_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported weights format. Provide a state_dict or a dict with key 'state_dict'.")

    cleaned = {}
    for k, v in state.items():
        cleaned[k[len("module.") :] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model



# Preprocessing (strict RGB ordering for optional .npy saving)
# Pipeline:
#   - detect + rotation normalize + crop
#   - resize 64x64
#   - grayscale (1ch)
#   - stretch back to 3ch
#   - convert to float [0,1] in RGB channel order (H,W,3)
#   - optionally save .npy (RGB)
#   - z-normalize with MEAN/STD
#   - feed into model via DataLoader
def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def crop_face_rgb_uint8(cropper: MTCNNFaceCropper, img_bgr: np.ndarray, face_index: int) -> Optional[np.ndarray]:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    results = cropper.process_pil(pil)
    if len(results) == 0 or face_index >= len(results):
        return None
    face_pil = results[face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
    return np.array(face_pil)  # RGB uint8, HxWx3


def rgb_uint8_to_gray3_rgb01(face_rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2GRAY)  # (H,W) uint8
    gray3 = np.stack([gray, gray, gray], axis=2)  # (H,W,3) uint8, RGB order (channels identical)
    return (gray3.astype(np.float32) / 255.0).clip(0.0, 1.0)  # float32, [0,1], RGB


def z_normalize_rgb01_to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    chw = rgb01.transpose(2, 0, 1).astype(np.float32, copy=False)  # 3xHxW (RGB)
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw)  # 3xHxW



# Dataset + DataLoader
@dataclass
class Sample:
    path: Path
    x: Optional[torch.Tensor]
    npy_path: Optional[Path]


class InferenceDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        cropper: MTCNNFaceCropper,
        face_index: int,
        save_npy: bool,
        npy_dir: Path,
    ):
        self.image_paths = image_paths
        self.cropper = cropper
        self.face_index = face_index
        self.save_npy = save_npy
        self.npy_dir = npy_dir

        if self.save_npy:
            self.npy_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        p = self.image_paths[idx]
        img_bgr = read_image_bgr(p)
        if img_bgr is None:
            return Sample(path=p, x=None, npy_path=None)

        face_rgb_u8 = crop_face_rgb_uint8(self.cropper, img_bgr, face_index=self.face_index)
        if face_rgb_u8 is None:
            return Sample(path=p, x=None, npy_path=None)

        # grayscale -> 3ch -> [0,1] in RGB order
        gray3_rgb01 = rgb_uint8_to_gray3_rgb01(face_rgb_u8)  # (64,64,3) float32, RGB

        saved_path: Optional[Path] = None
        if self.save_npy:
            saved_path = self.npy_dir / f"{p.stem}_face{self.face_index}.npy"
            np.save(str(saved_path), gray3_rgb01.astype(np.float32, copy=False))

        x = z_normalize_rgb01_to_tensor(gray3_rgb01)  # 3x64x64
        return Sample(path=p, x=x, npy_path=saved_path)


def collate_fn(samples: List[Sample]):
    paths = [s.path for s in samples]
    xs = [s.x for s in samples]
    npy_paths = [s.npy_path for s in samples]
    return paths, xs, npy_paths



# File iteration
def iter_images(images_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths



# Main
def main() -> int:
    ap = argparse.ArgumentParser()

    # NOTE: defaults are defined in code above; user can still override via CLI
    ap.add_argument("--images_dir", type=str, default=str(DEFAULT_IMAGES_DIR), help="Folder containing images")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Folder where CSV will be written")
    ap.add_argument("--out_name", type=str, default=DEFAULT_OUT_NAME, help="CSV filename")

    ap.add_argument("--model", type=str, default="model:Model", help='expects model.py with class Model in script folder')
    ap.add_argument("--weights", type=str, default=None, help="auto-picks .pt/.pth in script folder if omitted")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--face_index", type=int, default=0)
    ap.add_argument("--skip_no_face", action="store_true", help="skip images with no face (else write zeros)")

    # cropper knobs
    ap.add_argument("--crop_scale", type=float, default=1.15)
    ap.add_argument("--width_half", type=float, default=1.3)
    ap.add_argument("--min_prob", type=float, default=0.0)

    # optional npy saving
    ap.add_argument("--save_npy", action="store_true", help="save preprocessed npy (RGB, HxWx3, [0,1]) before z-norm")
    ap.add_argument("--npy_dir", type=str, default=str(DEFAULT_NPY_DIR), help="where to store npy files")

    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.out_name

    # sanity
    if set(CSV_CLASS_ORDER) != set(TRAIN_CLASS_ORDER):
        raise ValueError(
            "CSV_CLASS_ORDER and TRAIN_CLASS_ORDER must contain the same labels.\n"
            f"TRAIN: {TRAIN_CLASS_ORDER}\nCSV:   {CSV_CLASS_ORDER}"
        )

    device = torch.device(args.device)

    # init cropper
    cropper = MTCNNFaceCropper(
        keep_all=True,
        min_prob=args.min_prob,
        width_half=args.width_half,
        crop_scale=args.crop_scale,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # model + weights
    model = build_model(args.model, num_classes=len(TRAIN_CLASS_ORDER))

    if args.weights is not None:
        wpath = Path(args.weights).expanduser()
        if not wpath.is_absolute():
            wpath = (BASE_DIR / wpath).resolve()
    else:
        wpath = _pick_default_weights_path()

    if wpath is None or not wpath.exists():
        raise FileNotFoundError(
            "No weights found.\n"
            "Put a .pt/.pth file next to this script (recommended: model_state_dict.pt)\n"
            "or pass --weights /path/to/weights.pt"
        )

    model = load_weights(model, wpath, device)

    # collect images
    img_paths = iter_images(images_dir, recursive=args.recursive)
    if len(img_paths) == 0:
        print(f"[warning] no images found in: {images_dir}")
        return 0

    npy_dir = Path(args.npy_dir).expanduser().resolve()

    ds = InferenceDataset(
        img_paths,
        cropper=cropper,
        face_index=args.face_index,
        save_npy=args.save_npy,
        npy_dir=npy_dir,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    train_index = {name: i for i, name in enumerate(TRAIN_CLASS_ORDER)}
    header = ["filepath"] + CSV_CLASS_ORDER

    print(f"[info] images_dir: {images_dir}")
    print(f"[info] csv_out:    {out_csv}")
    print(f"[info] model:      {args.model}")
    print(f"[info] weights:    {wpath.name}")
    if args.save_npy:
        print(f"[info] npy_out:    {npy_dir}")
    print("[info] preprocessing: mtcnn->(rot)->crop->resize64 -> gray(1ch)->gray3(RGB)->[0,1]->(save npy)->z-norm->model")

    no_face = 0
    wrote = 0
    saved_npy = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for paths, xs, npy_paths in dl:
            valid_idx = [i for i, x in enumerate(xs) if x is not None]
            batch_probs = {}

            if args.save_npy:
                for npy_p in npy_paths:
                    if npy_p is not None:
                        saved_npy += 1

            if len(valid_idx) > 0:
                xb = torch.stack([xs[i] for i in valid_idx], dim=0).to(device)  # Nx3x64x64
                with torch.no_grad():
                    logits = model(xb)  # NxC
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
                for j, i in enumerate(valid_idx):
                    batch_probs[i] = probs[j]

            for i, p in enumerate(paths):
                prob = batch_probs.get(i, None)
                if prob is None:
                    no_face += 1
                    if args.skip_no_face:
                        continue
                    w.writerow([str(p)] + [0.0] * len(CSV_CLASS_ORDER))
                    wrote += 1
                else:
                    scores = [float(prob[train_index[label]]) for label in CSV_CLASS_ORDER]
                    w.writerow([str(p)] + scores)
                    wrote += 1

    print(f"[info] wrote rows: {wrote}")
    print(f"[info] no-face images: {no_face}")
    if args.save_npy:
        print(f"[info] saved npy files: {saved_npy}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
