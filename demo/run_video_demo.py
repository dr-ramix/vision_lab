#!/usr/bin/env python3
# vision_lab/demo/run_video_emotion_saliency.py
#
# Takes a video path, crops face(s) with your MTCNNFaceCropper, classifies emotion,
# highlights important regions (Grad-CAM if possible, otherwise input-gradient saliency),
# and saves a new video with prediction text + saliency overlay.
#
# Default output directory:
#   vision_lab/demo/kaggle_video_out
#
# Examples:
#   (1) Random weights (no --weights):
#   python demo/run_video_emotion_saliency.py --video_in demo/kaggle_videos/xyz.mp4 \
#       --model "fer.models.cnn_vanilla:ConvolutionalNetwork"
#
#   (2) With weights if available:
#   python demo/run_video_demo.py --video_in demo/kaggle_videos/3.mp4 \
#       --model "fer.models.cnn_resnet50:ResNet50FER" \
#       --weights training_output/runs/2026-01-06_13-14-27__resnet50__user-bargozideh__e30877/exports/model_state_dict.pt
#
#   (3) Explicit output path:
#   python demo/run_video_emotion_saliency.py --video_in demo/kaggle_videos/xyz.mp4 \
#       --model "fer.models.cnn_vanilla:ConvolutionalNetwork" \
#       --video_out demo/kaggle_video_out/xyz_overlay.mp4
#
# Notes:
# - Your cropper API used here: cropper.process_pil(PIL.Image) -> List[FaceCropResult] with .crop :contentReference[oaicite:0]{index=0}
# - Dataset stats are hardcoded to the values you provided.

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ensure imports work when run from repo root (vision_lab)
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../vision_lab
SRC_ROOT = REPO_ROOT / "main" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper  # :contentReference[oaicite:1]{index=1}
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import scale_cam_image
from fer.explainability.gradcam_custom import GradCAM

# --------------------------------------------------
# Fixed class order (must match your training)
# --------------------------------------------------
CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# --------------------------------------------------
# Dataset stats you provided
# --------------------------------------------------
MEAN = np.array([0.5461214492863451, 0.5461214492863451, 0.5461214492863451], dtype=np.float32)
STD = np.array([0.22092840651221893, 0.22092840651221893, 0.22092840651221893], dtype=np.float32)
TARGET_SIZE = (64, 64)  # (W,H)


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
    """
    Builds the model architecture only (weights handled separately).
    Swappable by changing --model and (optionally) --model_kwargs_json.
    """
    ModelCls = _import_from_string(model_spec)
    try:
        return ModelCls(num_classes=num_classes, **model_kwargs)
    except TypeError:
        # fallback: some models might not have num_classes kwarg
        return ModelCls(**model_kwargs)


def load_weights_if_available(model: nn.Module, weights_path: Optional[Path], device: torch.device) -> Tuple[nn.Module, bool]:
    """
    If weights_path exists -> load; else keep random weights.
    Returns (model, loaded_bool).
    """
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
        raise ValueError("Unsupported weights format. Provide state_dict or dict with key 'state_dict'.")

    cleaned = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, True


# ============================
# Preprocessing
# ============================
def bgr_to_rgb01(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def normalize_to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    """
    rgb01: HxWx3 float [0,1]
    returns: 1x3xHxW
    """
    chw = rgb01.transpose(2, 0, 1)
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw).unsqueeze(0)


# ============================
# Saliency (Grad-CAM or fallback)
# ============================

def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    last_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_layer = m
    return last_layer

def input_grad_saliency(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Fallback saliency: |dscore/dinput| aggregated over channels.
    Returns HxW float [0,1] at input resolution.
    """
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


# ============================
# Overlay helpers
# ============================
def overlay_heatmap_on_bgr(img_bgr: np.ndarray, heat01: np.ndarray, alpha: float) -> np.ndarray:
    heat_u8 = (heat01 * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0.0)


def put_text(img_bgr: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def paste_inset(dst_bgr: np.ndarray, inset_bgr: np.ndarray, top_left: Tuple[int, int]) -> None:
    x, y = top_left
    ih, iw = inset_bgr.shape[:2]
    H, W = dst_bgr.shape[:2]
    if x < 0 or y < 0 or x + iw > W or y + ih > H:
        return
    dst_bgr[y : y + ih, x : x + iw] = inset_bgr


# ============================
# Main
# ============================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_in", required=True, type=str, help="Input video path")
    ap.add_argument("--video_out", default=None, type=str, help="Output video path (optional)")
    ap.add_argument("--out_dir", default="demo/kaggle_video_out", type=str, help="Output directory if --video_out not set")

    ap.add_argument("--model", required=True, type=str, help='Model spec, e.g. "fer.models.cnn_vanilla:ConvolutionalNetwork"')
    ap.add_argument("--model_kwargs_json", default="{}", type=str, help='JSON dict for model kwargs, e.g. \'{"dropout":0.2}\'')

    ap.add_argument("--weights", default=None, type=str, help="Optional weights path (.pt). If missing/not found -> random weights.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.40, help="Heatmap overlay strength")
    ap.add_argument("--inset_scale", type=float, default=0.35, help="Inset width relative to frame width")
    ap.add_argument("--max_frames", type=int, default=-1, help="Debug: process only N frames; -1 = all")
    ap.add_argument("--face_index", type=int, default=0, help="Which detected face to use (0 = first)")
    args = ap.parse_args()

    video_in = (REPO_ROOT / args.video_in).resolve() if not os.path.isabs(args.video_in) else Path(args.video_in)
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    if args.video_out is not None:
        video_out = (REPO_ROOT / args.video_out).resolve() if not os.path.isabs(args.video_out) else Path(args.video_out)
        video_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = (REPO_ROOT / args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        video_out = out_dir / f"{video_in.stem}_overlay.mp4"

    weights_path = None
    if args.weights is not None:
        weights_path = (REPO_ROOT / args.weights).resolve() if not os.path.isabs(args.weights) else Path(args.weights)

    device = torch.device(args.device)

    model_kwargs = json.loads(args.model_kwargs_json)
    if not isinstance(model_kwargs, dict):
        raise ValueError("--model_kwargs_json must be a JSON object/dict")

    # Face cropper (your implementation)
    cropper = MTCNNFaceCropper(keep_all=True, min_prob=0.0)

    # Build + (optionally) load weights
    model = build_model(args.model, num_classes=len(CLASS_ORDER), model_kwargs=model_kwargs)
    model.eval()
    model, loaded = load_weights_if_available(model, weights_path, device)
    target_conv = find_last_conv_layer(model)

    if target_conv is None:
        print("[warning] No Conv2d layer found -> using input-gradient saliency")
        cam = None
    else:
        cam = GradCAM(
            model=model,
            target_layers=[target_conv],
            reshape_transform=None
        )
        print(f"[info] Using custom GradCAM on layer: {target_conv}")

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (W, H))

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

            # ---- face crop
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            results = cropper.process_pil(pil)
            if len(results) == 0 or args.face_index >= len(results):
                put_text(out, "no face", 15, 30)
                writer.write(out)
                continue

            face_pil = results[args.face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
            face_rgb = np.array(face_pil)  # 64x64x3 RGB uint8
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

            # ---- model input (match training: grayscale -> 3ch -> mean/std)
            gray_u8 = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)              # (64,64) uint8
            gray3_u8 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)         # (64,64,3) uint8

            gray3_01 = (gray3_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)   # float [0,1]
            x = normalize_to_tensor(gray3_01).to(device)

            # ---- predict (needs grads for saliency)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_name = CLASS_ORDER[pred_idx]
            pred_conf = float(probs[pred_idx].item())

            # ---- saliency
            if cam is not None:
                targets = [ClassifierOutputTarget(pred_idx)]

                cam_map = cam(
                    input_tensor=x,
                    targets=targets
                )[0]  # shape: HxW (normalized)

                cam_map = scale_cam_image(cam_map)
                heat64 = cv2.resize(cam_map, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

                saliency_kind = "Grad-CAM"
            else:
                sal = input_grad_saliency(model, x, pred_idx)  
                heat64 = sal.astype(np.float32)
                saliency_kind = "Input-Grad"

            face_overlay = overlay_heatmap_on_bgr(face_bgr, heat64, alpha=float(args.alpha))

            # ---- inset
            inset_w = int(W * float(args.inset_scale))
            inset_h = inset_w  # square
            inset = cv2.resize(face_overlay, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (255, 255, 255), 2)

            paste_inset(out, inset, (15, 55))

            # ---- texts
            put_text(out, f"{pred_name}  {pred_conf:.2f}", 15, 30)
            put_text(out, f"{saliency_kind} | weights={'loaded' if loaded else 'random'}", 15, 50)

            writer.write(out)

    finally:
        cap.release()
        writer.release()

    print(f"[saved] {video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
