from __future__ import annotations
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent                     # .../vision_lab/classification_model
REPO_ROOT = BASE_DIR.parent.parent                                    # .../vision_lab
MAIN_SRC = REPO_ROOT / "main" / "src"                          # .../vision_lab/main/src

if str(MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(MAIN_SRC))

from fer.models.registry import make_model  # noqa: E402


TRAIN_CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CSV_CLASS_ORDER = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]

MEAN = np.array([0.5368543512595557, 0.5368543512595557, 0.5368543512595557], dtype=np.float32)
STD = np.array([0.21882473736325334, 0.21882473736325334, 0.21882473736325334], dtype=np.float32)

TARGET_SIZE = (64, 64)  
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


HF_WEIGHTS: Dict[str, Dict[str, str]] = {
    "vgg19": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "vgg19/model_state_dict.pt",
    },
    "resnet18": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "resnet18/model_state_dict.pt",
    },
    "coatnetv3_small": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "coatnet_small/model_state_dict.pt",
    },
    "emocatnetsv2_base": {
        "repo_id": "lmuemonets/lmu_emonets",
        "filename": "emocatnetsv2base/model_state_dict.pt",
    },    
}


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
        img = np.array(pil_img)  
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
        raise ValueError(f"Unsupported checkpoint format for {model_name}")

    model.load_state_dict(_clean_state_dict(state_dict), strict=True)
    model.to(device).eval()
    return model



def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)

def crop_face_rgb_uint8(
    cropper: MTCNNFaceCropper,
    img_bgr: np.ndarray,
    face_index: int,
) -> Optional[np.ndarray]:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    results = cropper.process_pil(pil)
    if len(results) == 0 or face_index >= len(results):
        return None
    face_pil = results[face_index].crop.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR)
    return np.array(face_pil)  # RGB uint8, HxWx3


def rgb_uint8_to_gray3_rgb01(face_rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2GRAY) 
    gray3 = np.stack([gray, gray, gray], axis=2)           
    return (gray3.astype(np.float32) / 255.0).clip(0.0, 1.0)


def z_normalize_rgb01_to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    chw = rgb01.transpose(2, 0, 1).astype(np.float32, copy=False)  
    chw = (chw - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return torch.from_numpy(chw)  


@dataclass
class Sample:
    path: Path
    x: Optional[torch.Tensor]


class InferenceDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        cropper: MTCNNFaceCropper,
        face_index: int,
    ):
        self.image_paths = image_paths
        self.cropper = cropper
        self.face_index = face_index

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        p = self.image_paths[idx]
        img_bgr = read_image_bgr(p)
        if img_bgr is None:
            return Sample(path=p, x=None)

        face_rgb_u8 = crop_face_rgb_uint8(self.cropper, img_bgr, face_index=self.face_index)
        if face_rgb_u8 is None:
            return Sample(path=p, x=None)

        gray3_rgb01 = rgb_uint8_to_gray3_rgb01(face_rgb_u8)
        x = z_normalize_rgb01_to_tensor(gray3_rgb01)  
        return Sample(path=p, x=x)


def collate_fn(samples: List[Sample]):
    paths = [s.path for s in samples]
    xs = [s.x for s in samples]
    return paths, xs


def iter_images(images_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


# Ensembling
@torch.no_grad()
def softmax_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def ensemble_probs(per_model_probs: torch.Tensor, method: str) -> torch.Tensor:
    """
    per_model_probs: (M, C)
    returns: (C,)
    """
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Classify a folder of images into FER emotion scores using EmoCatNets v2/v3 (HF weights) + optional ensembling."
    )

    ap.add_argument("--images_dir", type=str, required=True, help="Input folder containing images.")
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(BASE_DIR / "classification_scores.csv"),
        help="Output CSV file path. Default: <script_dir>/classification_scores.csv",
    )

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--face_index", type=int, default=0)
    ap.add_argument("--skip_no_face", action="store_true", help="Skip images with no face (else write zeros).")

    ap.add_argument("--crop_scale", type=float, default=1.15)
    ap.add_argument("--width_half", type=float, default=1.3)
    ap.add_argument("--min_prob", type=float, default=0.0)

    allowed = sorted(HF_WEIGHTS.keys())
    ap.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble"] + allowed,
        help="Single model name or 'ensemble'. Default: ensemble.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["vgg19", "resnet18", "coatnetv3_small", "emocatnetsv2_base"],
        choices=allowed,
        help="Models used when --model ensemble.",
    )
    ap.add_argument(
        "--ensemble",
        type=str,
        default="majority",
        choices=["mean", "majority"],
        help="Ensembling strategy (only used when --model ensemble).",
    )
    ap.add_argument("--hf_revision", type=str, default=None)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    out_csv = Path(args.out_csv).expanduser()
    out_csv = out_csv.resolve() if out_csv.is_absolute() else (Path.cwd() / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if set(CSV_CLASS_ORDER) != set(TRAIN_CLASS_ORDER):
        raise ValueError(
            "CSV_CLASS_ORDER and TRAIN_CLASS_ORDER must contain the same labels.\n"
            f"TRAIN: {TRAIN_CLASS_ORDER}\nCSV:   {CSV_CLASS_ORDER}"
        )

    device = torch.device(args.device)

    cropper = MTCNNFaceCropper(
        keep_all=True,
        min_prob=args.min_prob,
        width_half=args.width_half,
        crop_scale=args.crop_scale,
        device=("cuda" if device.type == "cuda" else "cpu"),
    )

    img_paths = iter_images(images_dir, recursive=args.recursive)
    if len(img_paths) == 0:
        print(f"[warning] no images found in: {images_dir}")
        return 0

    ds = InferenceDataset(img_paths, cropper=cropper, face_index=args.face_index)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    num_classes = len(TRAIN_CLASS_ORDER)
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
        mode_label = f"ensemble({args.ensemble})[{','.join(model_names)}]"
    else:
        models = [
            load_model_from_hf(
                args.model,
                device=device,
                num_classes=num_classes,
                in_channels=in_channels,
                revision=args.hf_revision,
            )
        ]
        mode_label = args.model

    train_index = {name: i for i, name in enumerate(TRAIN_CLASS_ORDER)}
    header = ["filepath"] + CSV_CLASS_ORDER

    print(f"[info] images_dir: {images_dir}")
    print(f"[info] out_csv:    {out_csv}")
    print(f"[info] device:     {device}")
    print(f"[info] mode:       {mode_label}")
    print("[info] preprocessing: mtcnn->(rot)->crop->resize64 -> gray(1ch)->gray3(RGB)->[0,1]->z-norm->model")

    no_face = 0
    wrote = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for paths, xs in dl:
            valid_idx = [i for i, x in enumerate(xs) if x is not None]
            batch_probs: Dict[int, np.ndarray] = {}

            if len(valid_idx) > 0:
                xb = torch.stack([xs[i] for i in valid_idx], dim=0).to(device)  

                with torch.no_grad():
                    if args.model == "ensemble":
                        per_model = torch.stack([softmax_probs(m, xb) for m in models], dim=0)  
                        probs_list = []
                        for n in range(per_model.shape[1]):
                            probs_list.append(ensemble_probs(per_model[:, n, :], method=args.ensemble))
                        probs = torch.stack(probs_list, dim=0)  #
                    else:
                        probs = softmax_probs(models[0], xb) 

                probs_np = probs.detach().cpu().numpy().astype(np.float32)
                for j, i in enumerate(valid_idx):
                    batch_probs[i] = probs_np[j]

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
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
