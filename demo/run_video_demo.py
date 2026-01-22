#!/usr/bin/env python3
# demo/run_video_demo.py
#
# Video FER + XAI (Grad-CAM if possible else input-gradient),
# using Hugging Face weights + local `demo/src/models/*` registry (CSV-style).
#
# Layout expected:
#   demo/
#     run_video_demo.py
#     kaggle_videos/...
#     src/
#       models/
#         registry.py
#         cnn_resnet50.py
#         cnn_resnet101.py
#         emocatnetsv2_small.py
#         ...
#
# Requirements:
#   pip install torch torchvision facenet-pytorch pillow opencv-python numpy huggingface-hub pytorch-grad-cam
#
# Run:
#   python run_video_demo.py --video_dir kaggle_videos/3.mp4
#   python run_video_demo.py --video_dir kaggle_videos/3.mp4 --out_dir ./out
#   python run_video_demo.py --video_dir kaggle_videos/3.mp4 --model resnet50
#   python run_video_demo.py --video_dir kaggle_videos/3.mp4 --model ensemble --models resnet50 resnet101 emocatnetsv2_small
#
# Changes vs previous version:
# - NO inset box anymore.
# - XAI heatmap is overlaid DIRECTLY onto the detected face region in the original frame.

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ======================================================================================
# Path setup (LIKE your CSV script)
# ======================================================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # demo/
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # enables: from models.registry import make_model

from models.registry import make_model  # noqa: E402


# ======================================================================================
# Constants
# ======================================================================================

CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

MEAN = np.array([0.5461214492863451, 0.5461214492863451, 0.5461214492863451], dtype=np.float32)
STD = np.array([0.22092840651221893, 0.22092840651221893, 0.22092840651221893], dtype=np.float32)

TARGET_SIZE = (64, 64)  # (W, H)

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
# Face cropping with MTCNN (rotation-normalized) + returns original bbox
# ======================================================================================

@dataclass(frozen=True)
class FaceCropResult:
    face_index: int
    prob: Optional[float]
    crop: Image.Image
    eye_angle_before: float
    residual_angle_after: float
    used_rotation_sign: str  # "A(-eye_angle)" or "B(+eye_angle)"
    bbox_xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2 in ORIGINAL frame coords


class MTCNNFaceCropper:
    """
    Detect faces + eye landmarks, rotate by eye line (tries +/- eye_angle, picks best residual),
    then paper-crop using heuristic box around eyes (on rotated image).
    Also returns the original MTCNN bbox for overlaying heatmaps onto the original frame.
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

    def process_pil(self, img: Image.Image) -> List[FaceCropResult]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        boxes, probs, lms = self.mtcnn.detect(img, landmarks=True)
        if boxes is None or lms is None:
            return []

        if probs is None:
            probs = [1.0] * len(lms)

        results: List[FaceCropResult] = []

        for i, (box, prob, lm) in enumerate(zip(boxes, probs, lms)):
            if prob is not None and float(prob) < self.min_prob:
                continue

            left_eye = tuple(lm[0])
            right_eye = tuple(lm[1])
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            eye_angle = self._angle_deg(left_eye, right_eye)
            center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

            # Try both rotations; keep the one with smaller residual eye angle.
            rot_a, (l_a, r_a) = self._rotate_image_and_points(img, [left_eye, right_eye], -eye_angle, center)
            res_a = abs(self._angle_deg(l_a, r_a))

            rot_b, (l_b, r_b) = self._rotate_image_and_points(img, [left_eye, right_eye], +eye_angle, center)
            res_b = abs(self._angle_deg(l_b, r_b))

            if res_a <= res_b:
                rot_img, rot_left, rot_right = rot_a, l_a, r_a
                used = "A(-eye_angle)"
                residual = res_a
            else:
                rot_img, rot_left, rot_right = rot_b, l_b, r_b
                used = "B(+eye_angle)"
                residual = res_b

            # Paper crop heuristic around eyes after rotation
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

            crop = rot_img.crop((x1, y1, x2, y2))
            bbox_xyxy = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

            results.append(
                FaceCropResult(
                    face_index=i,
                    prob=None if prob is None else float(prob),
                    crop=crop,
                    eye_angle_before=float(eye_angle),
                    residual_angle_after=float(residual),
                    used_rotation_sign=used,
                    bbox_xyxy=bbox_xyxy,
                )
            )

        return results


# ======================================================================================
# HF model loading (LIKE CSV)
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
    if not strict:
        print("[warn] emocatnetsv2_small loaded with strict=False due to key-name mismatch")
        if missing:
            print("[warn] missing keys (showing up to 20):", missing[:20])
        if unexpected:
            print("[warn] unexpected keys (showing up to 20):", unexpected[:20])

    model.to(device).eval()
    return model


# ======================================================================================
# Preprocessing
# ======================================================================================

def normalize_to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    chw = rgb01.transpose(2, 0, 1)
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw.astype(np.float32, copy=False)).unsqueeze(0)


# ======================================================================================
# Saliency
# ======================================================================================

def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    last_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_layer = m
    return last_layer


def input_grad_saliency(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    g = x.grad[0]  # 3xHxW
    sal = g.abs().mean(dim=0)  # HxW
    sal = sal - sal.min()
    sal = sal / sal.max().clamp_min(1e-8)
    return sal.detach().cpu().numpy().astype(np.float32)


# ======================================================================================
# Overlay helpers
# ======================================================================================

def overlay_heatmap_on_bgr(img_bgr: np.ndarray, heat01: np.ndarray, alpha: float) -> np.ndarray:
    heat_u8 = (heat01 * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0.0)


def put_text(img_bgr: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ======================================================================================
# Ensembling
# ======================================================================================

@torch.no_grad()
def softmax_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def ensemble_probs(per_model_probs: torch.Tensor, method: str) -> torch.Tensor:
    method = method.lower().strip()

    if method == "mean":
        return per_model_probs.mean(dim=0)

    if method == "majority":
        votes = per_model_probs.argmax(dim=1)  # (M,)
        c = per_model_probs.shape[1]
        tally = torch.zeros((c,), dtype=torch.float32, device=per_model_probs.device)
        for v in votes:
            tally[int(v.item())] += 1.0
        return tally / tally.sum().clamp_min(1e-8)

    raise ValueError("Invalid ensemble method. Choose from: mean, majority")


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Video FER + saliency (HF weights) using local demo/src/models registry (CSV-style)."
    )

    ap.add_argument("--video_dir", required=True, type=str, help="Input video path (file).")
    ap.add_argument(
        "--out_dir",
        default=str(SCRIPT_DIR),
        type=str,
        help="Output directory (default: script folder).",
    )
    ap.add_argument("--video_out_name", default=None, type=str, help="Optional output filename (default: <stem>_overlay.mp4)")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--face_index", type=int, default=0)

    # Face cropper params
    ap.add_argument("--min_prob", type=float, default=0.0)
    ap.add_argument("--width_half", type=float, default=1.3)
    ap.add_argument("--crop_scale", type=float, default=1.15)

    # Models
    ap.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble"] + sorted(HF_WEIGHTS.keys()),
        help="Pick single model or 'ensemble' (default).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "resnet101", "emocatnetsv2_small"],
        choices=sorted(HF_WEIGHTS.keys()),
        help="Models used when --model ensemble.",
    )
    ap.add_argument(
        "--ensemble",
        type=str,
        default="majority",
        choices=["mean", "majority"],
        help="Ensembling strategy (only used when --model ensemble).",
    )
    ap.add_argument(
        "--saliency_model",
        type=str,
        default=None,
        choices=sorted(HF_WEIGHTS.keys()),
        help="Which model to use for saliency when ensembling (default: first in --models).",
    )
    ap.add_argument("--hf_revision", type=str, default=None)

    return ap.parse_args()


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    # Resolve input relative to SCRIPT_DIR (demo/) if not absolute
    video_in = Path(args.video_dir).expanduser()
    if not video_in.is_absolute():
        video_in = (SCRIPT_DIR / video_in).resolve()
    else:
        video_in = video_in.resolve()
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    # Output dir relative to SCRIPT_DIR by default
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (SCRIPT_DIR / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video_out_name is None:
        video_out = out_dir / f"{video_in.stem}_overlay.mp4"
    else:
        video_out = out_dir / args.video_out_name

    device = torch.device(args.device)

    cropper = MTCNNFaceCropper(
        device=("cuda" if device.type == "cuda" else "cpu"),
        keep_all=True,
        min_prob=args.min_prob,
        width_half=args.width_half,
        crop_scale=args.crop_scale,
    )

    num_classes = len(CLASS_ORDER)
    in_channels = 3

    # Load model(s)
    if args.model == "ensemble":
        model_names = list(args.models)
        models: List[nn.Module] = [
            load_model_from_hf(
                name,
                device=device,
                num_classes=num_classes,
                in_channels=in_channels,
                revision=args.hf_revision,
            )
            for name in model_names
        ]
        sal_name = args.saliency_model or model_names[0]
        saliency_model = load_model_from_hf(
            sal_name,
            device=device,
            num_classes=num_classes,
            in_channels=in_channels,
            revision=args.hf_revision,
        )
        mode_label = f"ensemble({args.ensemble})"
    else:
        models = []
        saliency_model = load_model_from_hf(
            args.model,
            device=device,
            num_classes=num_classes,
            in_channels=in_channels,
            revision=args.hf_revision,
        )
        mode_label = args.model

    # Saliency: Grad-CAM if possible, else input-grad
    try:
        from pytorch_grad_cam import GradCAM as PTGradCAM  # noqa: E402
    except Exception:
        PTGradCAM = None  # type: ignore

    target_conv = find_last_conv_layer(saliency_model)
    if PTGradCAM is None or target_conv is None:
        cam = None
        if PTGradCAM is None:
            print("[warn] pytorch-grad-cam not available -> using input-gradient saliency")
        else:
            print("[warn] No Conv2d layer found -> using input-gradient saliency")
    else:
        cam = PTGradCAM(model=saliency_model, target_layers=[target_conv])
        print(f"[info] Using Grad-CAM on layer: {target_conv}")

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (W, H))

    print(f"[info] video_in:  {video_in}")
    print(f"[info] video_out: {video_out}")
    print(f"[info] device:    {device}")
    print(f"[info] mode:      {mode_label}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            out = frame_bgr.copy()

            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            results = cropper.process_pil(pil)
            if len(results) == 0 or args.face_index >= len(results):
                put_text(out, "no face", 15, 30)
                writer.write(out)
                continue

            # Original frame bbox for overlay placement
            bx1, by1, bx2, by2 = results[args.face_index].bbox_xyxy
            x1 = int(max(0, min(W - 1, round(bx1))))
            y1 = int(max(0, min(H - 1, round(by1))))
            x2 = int(max(0, min(W,     round(bx2))))
            y2 = int(max(0, min(H,     round(by2))))
            if x2 <= x1 or y2 <= y1:
                put_text(out, "bad bbox", 15, 30)
                writer.write(out)
                continue

            # Model input face crop (rotation-normalized paper crop)
            face_pil = results[args.face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
            face_rgb_u8 = np.array(face_pil)

            # grayscale -> 3ch -> [0,1] -> z-norm
            gray_u8 = cv2.cvtColor(face_rgb_u8, cv2.COLOR_RGB2GRAY)
            gray3_u8 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
            gray3_01 = (gray3_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
            x = normalize_to_tensor(gray3_01).to(device)  # 1x3x64x64

            # predict
            if args.model == "ensemble":
                with torch.no_grad():
                    per = torch.stack([softmax_probs(m, x)[0] for m in models], dim=0)  # (M,C)
                    probs = ensemble_probs(per, method=args.ensemble)  # (C,)
                pred_idx = int(torch.argmax(probs).item())
                pred_conf = float(probs[pred_idx].item())
            else:
                with torch.no_grad():
                    probs = softmax_probs(saliency_model, x)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_conf = float(probs[pred_idx].item())

            pred_name = CLASS_ORDER[pred_idx]

            # saliency map at 64x64
            if cam is not None:
                cam_map = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]  # HxW
                cam_map = cam_map.astype(np.float32)
                cam_map = cam_map - cam_map.min()
                cam_map = cam_map / (cam_map.max() + 1e-8)
                heat64 = cv2.resize(cam_map, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                saliency_kind = "Grad-CAM"
            else:
                heat64 = input_grad_saliency(saliency_model, x, pred_idx)
                heat64 = cv2.resize(heat64, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                saliency_kind = "Input-Grad"

            # Overlay DIRECTLY onto face bbox in the ORIGINAL frame
            roi = out[y1:y2, x1:x2]  # BGR
            heat_roi = cv2.resize(heat64, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = overlay_heatmap_on_bgr(roi, heat_roi, alpha=float(args.alpha))

            # Text near face
            put_text(out, f"{pred_name}  {pred_conf:.2f}", x1, max(20, y1 - 10))
            put_text(out, f"{saliency_kind} | {mode_label}", x1, min(H - 5, y2 + 20))

            writer.write(out)

    finally:
        cap.release()
        writer.release()

    print(f"[saved] {video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
