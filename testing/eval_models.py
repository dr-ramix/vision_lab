# ausführen mit: python eval_models.py --images-dir ckplus --out-csv results_ckplus.csv
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset

# ======================================================================================
# Project path setup
# Keep this script at repo root OR adjust PROJECT_ROOT accordingly.
# If you place it in vision_lab/testing/, PROJECT_ROOT should be 2 levels up.
# ======================================================================================

# If script is in: vision_lab/vision_lab/testing/eval_models.py
# then repo root is:      vision_lab/vision_lab
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))  # enables: from models.registry import make_model

from fer.models.registry import make_model  # noqa: E402

# ======================================================================================
# Constants / configuration
# ======================================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Training class order (must match model training)
TRAIN_CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASS_ORDER)}

# Training stats (computed on training preprocessing output in [0,1])
MEAN = np.array([0.5426446906981507, 0.5426446906981507, 0.5426446906981507], dtype=np.float32)
STD = np.array([0.22369591629278052, 0.22369591629278052, 0.22369591629278052], dtype=np.float32)

TARGET_SIZE = (64, 64)  # (W, H)

# Hugging Face mapping 
HF_WEIGHTS: Dict[str, Dict[str, str]] = {
    "resnet50": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "resnet50/model_state_dict.pt",
    },
    "resnet101": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "resnet101/model_state_dict.pt",
    },
    "emocatnetsv2_small": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "emocatnetsv2_small/model_state_dict.pt",
    },
}

# ======================================================================================
# Face cropping with MTCNN (rotation-normalized)
# ======================================================================================
@dataclass(frozen=True)
class FaceCrop:
    crop: Image.Image


class MTCNNFaceCropper:
    """
    Detects faces, aligns by eye angle (tries +/- eye_angle, picks best residual),
    then crops using a heuristic box around the eyes.
    """

    def __init__(
        self,
        *,
        device: str,
        keep_all: bool = True,
        min_prob: float = 0.0,
        width_half: float = 1.3,
        crop_scale: float = 1.15,
    ) -> None:
        self.keep_all = keep_all
        self.min_prob = float(min_prob)
        self.width_half = float(width_half)
        self.crop_scale = float(crop_scale)
        self.device = device
        self.mtcnn = MTCNN(keep_all=self.keep_all, device=self.device)

    @staticmethod
    def _rotate_image_and_points(
        img: Image.Image,
        points: List[Tuple[float, float]],
        angle_deg: float,
        center_xy: Tuple[float, float],
    ) -> Tuple[Image.Image, List[Tuple[float, float]]]:
        arr = np.array(img)  # RGB
        h, w = arr.shape[:2]
        cx, cy = center_xy

        m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            arr,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        pts = np.array(points, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
        pts_rot = (m @ pts_h.T).T

        return Image.fromarray(rotated), [tuple(p) for p in pts_rot]

    @staticmethod
    def _angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def crop_faces(self, img: Image.Image) -> List[FaceCrop]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        _boxes, probs, lms = self.mtcnn.detect(img, landmarks=True)
        if lms is None:
            return []

        if probs is None:
            probs = [1.0] * len(lms)

        results: List[FaceCrop] = []

        for prob, lm in zip(probs, lms):
            if prob is not None and float(prob) < self.min_prob:
                continue

            left_eye = tuple(lm[0])
            right_eye = tuple(lm[1])
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            eye_angle = self._angle_deg(left_eye, right_eye)
            center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

            rot_a, (l_a, r_a) = self._rotate_image_and_points(img, [left_eye, right_eye], -eye_angle, center)
            res_a = abs(self._angle_deg(l_a, r_a))

            rot_b, (l_b, r_b) = self._rotate_image_and_points(img, [left_eye, right_eye], +eye_angle, center)
            res_b = abs(self._angle_deg(l_b, r_b))

            if res_a <= res_b:
                rot_img, rot_left, rot_right = rot_a, l_a, r_a
            else:
                rot_img, rot_left, rot_right = rot_b, l_b, r_b

            mx = (rot_left[0] + rot_right[0]) / 2.0
            my = (rot_left[1] + rot_right[1]) / 2.0
            alpha = math.hypot(rot_right[0] - mx, rot_right[1] - my) * self.crop_scale

            x1 = mx - self.width_half * alpha
            x2 = mx + self.width_half * alpha
            y1 = my - 1.3 * alpha
            y2 = my + 3.2 * alpha

            w, h = rot_img.size
            x1 = self._clamp(x1, 0, w)
            x2 = self._clamp(x2, 0, w)
            y1 = self._clamp(y1, 0, h)
            y2 = self._clamp(y2, 0, h)

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(FaceCrop(crop=rot_img.crop((x1, y1, x2, y2))))

        return results


# ======================================================================================
# Preprocessing
# ======================================================================================
def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def preprocess_face_to_tensor(
    cropper: MTCNNFaceCropper,
    img_bgr: np.ndarray,
    *,
    face_index: int,
) -> Optional[torch.Tensor]:
    """
    Pipeline:
      - BGR -> RGB PIL
      - detect + align + crop
      - resize to 64x64
      - grayscale -> 3ch (RGB order)
      - scale to [0,1]
      - z-norm with MEAN/STD (per-channel)
      - returns torch tensor: 3x64x64
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    faces = cropper.crop_faces(pil)
    if not faces or face_index >= len(faces):
        return None

    face_rgb = faces[face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
    face_rgb_u8 = np.array(face_rgb)  # HxWx3, RGB uint8

    gray = cv2.cvtColor(face_rgb_u8, cv2.COLOR_RGB2GRAY)  # HxW
    gray3 = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.0  # HxWx3, [0,1]

    chw = gray3.transpose(2, 0, 1)  # 3xHxW
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)

    return torch.from_numpy(chw.astype(np.float32, copy=False))


# ======================================================================================
# Labeled dataset (folder-per-class)
# ======================================================================================
@dataclass(frozen=True)
class Sample:
    path: Path
    y: int
    x: Optional[torch.Tensor]


def iter_labeled_images(images_dir: Path, *, recursive: bool) -> List[Tuple[Path, int]]:
    """
    Expects:
      images_dir/<class_name>/*.jpg
    where <class_name> in TRAIN_CLASS_ORDER
    """
    items: List[Tuple[Path, int]] = []
    for cls in TRAIN_CLASS_ORDER:
        cls_dir = images_dir / cls
        if not cls_dir.exists():
            continue
        paths = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
        for p in paths:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, CLASS_TO_IDX[cls]))
    return sorted(items, key=lambda t: str(t[0]))


class LabeledInferenceDataset(Dataset):
    def __init__(
        self,
        labeled_paths: Sequence[Tuple[Path, int]],
        *,
        cropper: MTCNNFaceCropper,
        face_index: int,
    ) -> None:
        self.items = list(labeled_paths)
        self.cropper = cropper
        self.face_index = face_index

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        path, y = self.items[idx]
        img_bgr = read_image_bgr(path)
        if img_bgr is None:
            return Sample(path=path, y=y, x=None)
        x = preprocess_face_to_tensor(self.cropper, img_bgr, face_index=self.face_index)
        return Sample(path=path, y=y, x=x)


def collate_samples(samples: List[Sample]) -> Tuple[List[Path], torch.Tensor, List[Optional[torch.Tensor]]]:
    paths = [s.path for s in samples]
    ys = torch.tensor([s.y for s in samples], dtype=torch.long)
    xs = [s.x for s in samples]
    return paths, ys, xs


# ======================================================================================
# Model loading from Hugging Face
# ======================================================================================
def _remap_emocatnetsv2_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("stem.0.", "stem.conv1.")
        nk = nk.replace("stem.2.", "stem.conv2.")
        nk = nk.replace("stem.3.", "stem.norm.")
        out[nk] = v
    return out


def _clean_state_dict(state: Dict[str, torch.Tensor], model_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        sd[k[7:] if k.startswith("module.") else k] = v
    if model_name == "emocatnetsv2_small":
        sd = _remap_emocatnetsv2_keys(sd)
    return sd


def load_model_from_hf(
    model_name: str,
    *,
    device: torch.device,
    num_classes: int,
    in_channels: int,
    revision: Optional[str] = None,
) -> nn.Module:
    if model_name not in HF_WEIGHTS:
        raise ValueError(f"Unknown model '{model_name}'. Allowed: {sorted(HF_WEIGHTS.keys())}")

    model = make_model(model_name, num_classes=num_classes, in_channels=in_channels)

    meta = HF_WEIGHTS[model_name]
    ckpt_path = hf_hub_download(
        repo_id=meta["repo_id"],
        filename=meta["filename"],
        revision=revision,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format for {model_name} (expected dict or dict['state_dict']).")

    strict = (model_name != "emocatnetsv2_small")
    missing, unexpected = model.load_state_dict(
        _clean_state_dict(state_dict, model_name=model_name),
        strict=strict,
    )

    if model_name == "emocatnetsv2_small":
        # Silence expected missing keys for older checkpoints
        allowed_missing = {"stem.skip.weight"}
        real_missing = sorted(set(missing) - allowed_missing)
        if real_missing:
            print("[warn] emocatnetsv2_small missing keys (showing up to 20):", real_missing[:20])
        if unexpected:
            print("[warn] emocatnetsv2_small unexpected keys (showing up to 20):", unexpected[:20])
        if not strict:
            print("[warn] emocatnetsv2_small loaded with strict=False due to key-name mismatch")
    else:
        # For other models, keep strict=True so mismatches raise; no warnings needed
        pass

    model.to(device).eval()
    return model


# ======================================================================================
# Accuracy evaluation
# ======================================================================================
@torch.no_grad()
def eval_model_accuracy(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    skip_no_face: bool,
) -> Tuple[float, int, int, int]:
    """
    Returns:
      accuracy, n_total, n_used, n_no_face
    where:
      n_total = all images in dataset
      n_used  = images actually evaluated (face detected / tensor exists), unless skip_no_face=False -> still not used
      n_no_face = images where preprocessing produced no tensor (no face or load fail)
    """
    n_total = 0
    n_used = 0
    n_no_face = 0
    n_correct = 0

    for _paths, ys, xs in loader:
        n_total += len(xs)

        valid = [i for i, x in enumerate(xs) if x is not None]
        n_no_face += (len(xs) - len(valid))

        if valid:
            xb = torch.stack([xs[i] for i in valid], dim=0).to(device)
            yb = ys[valid].to(device)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1)

            n_used += len(valid)
            n_correct += int((pred == yb).sum().item())

    denom = n_used if skip_no_face else n_total
    if denom == 0:
        return 0.0, n_total, n_used, n_no_face

    if skip_no_face:
        acc = n_correct / max(1, n_used)
    else:
        acc = n_correct / max(1, n_total)

    return float(acc), n_total, n_used, n_no_face


# ======================================================================================
# CLI
# ======================================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HF FER models: accuracy per model on labeled folder dataset.")

    p.add_argument("--images-dir", type=str, required=True, help="Dataset root: contains class subfolders.")
    p.add_argument("--recursive", action="store_true", help="Scan class folders recursively.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Face detection / cropping
    p.add_argument("--face-index", type=int, default=0, help="Which detected face to use (0 = first).")
    p.add_argument("--min-prob", type=float, default=0.0, help="Minimum MTCNN face probability.")
    p.add_argument("--width-half", type=float, default=1.3, help="Crop width half-factor.")
    p.add_argument("--crop-scale", type=float, default=1.15, help="Crop scale multiplier.")
    p.add_argument("--skip-no-face", action="store_true", help="Ignore no-face images in accuracy denominator.")

    # Models
    p.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "resnet101", "emocatnetsv2_small"],
        choices=sorted(HF_WEIGHTS.keys()),
        help="Models to evaluate.",
    )
    p.add_argument("--hf-revision", type=str, default=None, help="Optional Hugging Face revision/tag/commit.")

    # Output
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write summary CSV (model,accuracy,n_total,n_used,n_no_face).",
    )

    return p.parse_args()


# ======================================================================================
# Main
# ======================================================================================
def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")

    labeled = iter_labeled_images(images_dir, recursive=args.recursive)
    if not labeled:
        raise RuntimeError(
            "No labeled images found.\n"
            "Expected structure:\n"
            "  images_dir/anger/*.jpg\n"
            "  images_dir/disgust/*.jpg\n"
            "  ...\n"
            f"Classes: {TRAIN_CLASS_ORDER}\n"
            f"Got images-dir: {images_dir}"
        )

    device = torch.device(args.device)

    cropper = MTCNNFaceCropper(
        device=("cuda" if device.type == "cuda" else "cpu"),
        keep_all=True,
        min_prob=args.min_prob,
        width_half=args.width_half,
        crop_scale=args.crop_scale,
    )

    ds = LabeledInferenceDataset(labeled, cropper=cropper, face_index=args.face_index)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )

    num_classes = len(TRAIN_CLASS_ORDER)
    in_channels = 3

    print(f"[info] images_dir:    {images_dir}")
    print(f"[info] device:        {device}")
    print(f"[info] models:        {args.models}")
    print(f"[info] skip_no_face:  {args.skip_no_face}")
    print(f"[info] n_images:      {len(ds)}")

    results: List[Tuple[str, float, int, int, int]] = []

    for name in args.models:
        model = load_model_from_hf(
            name,
            device=device,
            num_classes=num_classes,
            in_channels=in_channels,
            revision=args.hf_revision,
        )

        acc, n_total, n_used, n_no_face = eval_model_accuracy(
            model,
            loader,
            device=device,
            skip_no_face=args.skip_no_face,
        )

        denom = n_used if args.skip_no_face else n_total
        print(
            f"[result] {name:18s}  acc={acc*100:6.2f}%  "
            f"denom={denom}  total={n_total}  used={n_used}  no_face={n_no_face}"
        )
        results.append((name, acc, n_total, n_used, n_no_face))

    if args.out_csv is not None:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "accuracy", "n_total", "n_used", "n_no_face"])
            for name, acc, n_total, n_used, n_no_face in results:
                w.writerow([name, f"{acc:.6f}", n_total, n_used, n_no_face])
        print(f"[info] wrote summary csv: {out_path}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
