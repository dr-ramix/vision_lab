from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MAIN_SRC = REPO_ROOT / "main" / "src"

if str(MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(MAIN_SRC))

from fer.models.registry import make_model  # noqa: E402

CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

MEAN = np.array([0.5368543512595557, 0.5368543512595557, 0.5368543512595557], dtype=np.float32)
STD = np.array([0.21882473736325334, 0.21882473736325334, 0.21882473736325334], dtype=np.float32)

TARGET_SIZE = (64, 64)

HF_WEIGHTS: Dict[str, Dict[str, str]] = {
    "emocatnetsv2_nano": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "emocatnetsv2_nano/model_state_dict.pt",
    },
    "emocatnetsv3_nano": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "emocatnetsv3_nano/model_state_dict.pt",
    },
}


@dataclass(frozen=True)
class FaceCropResult:
    face_index: int
    prob: Optional[float]
    crop: Image.Image
    eye_angle_before: float
    residual_angle_after: float
    used_rotation_sign: str
    bbox_xyxy: Tuple[float, float, float, float]


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

        for i, (box, prob, lm) in enumerate(zip(boxes, probs, lms)):
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

            rot_img_a, (l_a, r_a) = self._rotate_image_and_points_cv2(
                img, [left_eye, right_eye], angle_deg=-eye_angle, center_xy=center
            )
            res_a = abs(self._angle_from_pts(l_a, r_a))

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

            mx = (rot_left[0] + rot_right[0]) / 2.0
            my = (rot_left[1] + rot_right[1]) / 2.0
            alpha = math.hypot(rot_right[0] - mx, rot_right[1] - my) * self.crop_scale

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
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )

        return results


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


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
    ckpt_path = hf_hub_download(repo_id=meta["repo_id"], filename=meta["filename"], revision=revision)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format for {model_name}")

    model.load_state_dict(_clean_state_dict(state_dict), strict=True)
    model.to(device).eval()
    return model


def normalize_to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    chw = rgb01.transpose(2, 0, 1)
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw.astype(np.float32, copy=False)).unsqueeze(0)


def input_grad_saliency(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    g = x.grad[0]
    sal = g.abs().mean(dim=0)
    sal = sal - sal.min()
    sal = sal / sal.max().clamp_min(1e-8)
    return sal.detach().cpu().numpy().astype(np.float32)


def overlay_heatmap_on_bgr(img_bgr: np.ndarray, heat01: np.ndarray, alpha: float) -> np.ndarray:
    heat_u8 = (heat01 * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0.0)


def put_text(img_bgr: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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
        votes = per_model_probs.argmax(dim=1)
        c = per_model_probs.shape[1]
        tally = torch.zeros((c,), dtype=torch.float32, device=per_model_probs.device)
        for v in votes:
            tally[int(v.item())] += 1.0
        return tally / tally.sum().clamp_min(1e-8)

    raise ValueError("Invalid ensemble method. Choose from: mean, majority")


def pick_cam_target_layer(model: nn.Module, explicit_name: Optional[str] = None) -> Tuple[str, nn.Module]:
    if explicit_name:
        for name, m in model.named_modules():
            if name == explicit_name:
                return name, m
        raise ValueError(f"--cam_layer '{explicit_name}' not found in model.named_modules().")

    if hasattr(model, "stage3"):
        stage3 = getattr(model, "stage3")
        if isinstance(stage3, nn.Sequential) and len(stage3) > 0:
            last_block = stage3[-1]
            if hasattr(last_block, "depthwise_conv") and isinstance(getattr(last_block, "depthwise_conv"), nn.Conv2d):
                return "stage3[-1].depthwise_conv", getattr(last_block, "depthwise_conv")
            for n, m in last_block.named_modules():
                if isinstance(m, nn.Conv2d):
                    return f"stage3[-1].{n}", m

    last_name = None
    last_conv = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name = name
            last_conv = m
    if last_conv is None or last_name is None:
        raise ValueError("No Conv2d found for Grad-CAM target.")
    return last_name, last_conv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Video FER + saliency overlay (HF weights) using repo-internal registry + inline MTCNN cropper."
    )

    ap.add_argument("--video_path", required=True, type=str, help="Input video path (file).")
    ap.add_argument("--out_dir", default=str(SCRIPT_DIR), type=str, help="Output directory (default: demo/).")
    ap.add_argument("--video_out_name", default=None, type=str)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--face_index", type=int, default=0)

    ap.add_argument("--min_prob", type=float, default=0.0)
    ap.add_argument("--width_half", type=float, default=1.3)
    ap.add_argument("--crop_scale", type=float, default=1.15)

    allowed = sorted(HF_WEIGHTS.keys())
    ap.add_argument("--model", type=str, default="ensemble", choices=["ensemble"] + allowed)
    ap.add_argument("--models", nargs="+", default=["emocatnetsv3_nano", "emocatnetsv2_nano"], choices=allowed)
    ap.add_argument("--ensemble", type=str, default="majority", choices=["mean", "majority"])
    ap.add_argument("--saliency_model", type=str, default=None, choices=allowed)
    ap.add_argument("--hf_revision", type=str, default=None)

    ap.add_argument("--cam_layer", type=str, default=None, help="Explicit model module name for CAM target (optional).")

    #prediction pacing / smoothing
    ap.add_argument("--predict_every", type=int, default=5, help="Update prediction + saliency every N frames.")
    ap.add_argument("--smooth_alpha", type=float, default=0.7, help="EMA smoothing for probs (0 disables).")
    ap.add_argument("--smooth_cam_alpha", type=float, default=0.6, help="EMA smoothing for CAM/heatmap (0 disables).")
    ap.add_argument(
        "--hold_label_frames",
        type=int,
        default=0,
        help="Hold the displayed label for at least K frames (0 disables).",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    video_in = Path(args.video_path).expanduser().resolve()
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    out_dir = Path(args.out_dir).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out = out_dir / (args.video_out_name if args.video_out_name else f"{video_in.stem}_overlay.mp4")

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

    if args.model == "ensemble":
        model_names = list(args.models)
        if len(model_names) == 0:
            raise ValueError("--models must contain at least one model for ensembling.")

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

    # Grad-CAM
    try:
        from pytorch_grad_cam import GradCAM as PTGradCAM  # noqa: E402
    except Exception:
        PTGradCAM = None

    if PTGradCAM is None:
        cam = None
        print("[warn] pytorch-grad-cam not available -> using input-gradient saliency")
    else:
        layer_name, target_layer = pick_cam_target_layer(saliency_model, explicit_name=args.cam_layer)
        cam = PTGradCAM(model=saliency_model, target_layers=[target_layer])
        print(f"[info] Using Grad-CAM target: {layer_name} ({type(target_layer).__name__})")

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[info] video_in:  {video_in}")
    print(f"[info] out_dir:   {out_dir}")
    print(f"[info] video_out: {video_out}")
    print(f"[info] device:    {device}")
    print(f"[info] mode:      {mode_label}")
    print(f"[info] predict_every: {args.predict_every} frames")
    print(f"[info] smooth_alpha: {args.smooth_alpha} | smooth_cam_alpha: {args.smooth_cam_alpha}")

    #state for pacing / smoothing
    last_probs: Optional[np.ndarray] = None  
    last_heat64: Optional[np.ndarray] = None  
    last_pred_idx: Optional[int] = None
    last_pred_conf: float = 0.0
    hold_counter = 0

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

            face_res = results[args.face_index]
            bx1, by1, bx2, by2 = face_res.bbox_xyxy

            x1 = int(max(0, min(W - 1, round(bx1))))
            y1 = int(max(0, min(H - 1, round(by1))))
            x2 = int(max(0, min(W, round(bx2))))
            y2 = int(max(0, min(H, round(by2))))
            if x2 <= x1 or y2 <= y1:
                put_text(out, "bad bbox", 15, 30)
                writer.write(out)
                continue

            face_crop_bgr = frame_bgr[y1:y2, x1:x2]
            if face_crop_bgr.size == 0:
                put_text(out, "empty crop", 15, 30)
                writer.write(out)
                continue

            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_crop_rgb).resize(TARGET_SIZE, resample=Image.BILINEAR)
            face_rgb_u8 = np.array(face_pil)

            gray_u8 = cv2.cvtColor(face_rgb_u8, cv2.COLOR_RGB2GRAY)
            gray3_u8 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
            gray3_01 = (gray3_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
            x = normalize_to_tensor(gray3_01).to(device)

            do_update = (last_probs is None) or (args.predict_every <= 1) or ((frame_idx % args.predict_every) == 0)

            if do_update:
                if args.model == "ensemble":
                    per = torch.stack([softmax_probs(m, x)[0] for m in models], dim=0)
                    probs_t = ensemble_probs(per, method=args.ensemble)  # (C,)
                else:
                    probs_t = softmax_probs(saliency_model, x)[0]  # (C,)

                probs_np = probs_t.detach().float().cpu().numpy()

                # Smooth probabilities (EMA)
                if last_probs is None or args.smooth_alpha <= 0.0:
                    smoothed_probs = probs_np
                else:
                    a = float(args.smooth_alpha)
                    smoothed_probs = a * last_probs + (1.0 - a) * probs_np

                # Candidate prediction
                cand_idx = int(np.argmax(smoothed_probs))
                cand_conf = float(smoothed_probs[cand_idx])

                if last_pred_idx is None:
                    last_pred_idx = cand_idx
                    last_pred_conf = cand_conf
                    hold_counter = args.hold_label_frames
                else:
                    if args.hold_label_frames > 0 and hold_counter > 0:
                        hold_counter -= 1
                    else:
                        if cand_idx != last_pred_idx:
                            last_pred_idx = cand_idx
                            hold_counter = args.hold_label_frames
                        last_pred_conf = cand_conf

                last_probs = smoothed_probs

                # Saliency map (compute only on update frames)
                if cam is not None:
                    cam_map = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(last_pred_idx))])[0]
                    cam_map = cam_map.astype(np.float32)
                    cam_map = cam_map - cam_map.min()
                    cam_map = cam_map / (cam_map.max() + 1e-8)
                    heat64_new = cv2.resize(cam_map, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                    saliency_kind = "Grad-CAM"
                else:
                    heat64_new = input_grad_saliency(saliency_model, x, int(last_pred_idx))
                    heat64_new = cv2.resize(heat64_new, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                    saliency_kind = "Input-Grad"

                # Smooth heatmap (EMA)
                if last_heat64 is None or args.smooth_cam_alpha <= 0.0:
                    last_heat64 = heat64_new
                else:
                    a = float(args.smooth_cam_alpha)
                    last_heat64 = a * last_heat64 + (1.0 - a) * heat64_new

                # Clamp numeric stability
                last_heat64 = np.clip(last_heat64, 0.0, 1.0)

                last_saliency_kind = saliency_kind
            else:
                # Reuse cached outputs
                if last_pred_idx is None:
                    put_text(out, "no pred", 15, 30)
                    writer.write(out)
                    continue
                last_saliency_kind = "Grad-CAM" if cam is not None else "Input-Grad"

            pred_name = CLASS_ORDER[int(last_pred_idx)]
            pred_conf = float(last_pred_conf)

            # Overlay into bbox ROI using cached heatmap
            heat64 = last_heat64 if last_heat64 is not None else np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)
            roi = out[y1:y2, x1:x2]
            heat_roi = cv2.resize(heat64, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = overlay_heatmap_on_bgr(roi, heat_roi, alpha=float(args.alpha))

            put_text(out, f"{pred_name}  {pred_conf:.2f}", x1, max(20, y1 - 10))
            put_text(out, f"{last_saliency_kind} | {mode_label}", x1, min(H - 5, y2 + 20))

            writer.write(out)

    finally:
        cap.release()
        writer.release()

    print(f"[saved] {video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

