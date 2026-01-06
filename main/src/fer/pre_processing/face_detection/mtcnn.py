# src/fer/pre_processing/mtcnn_cropper.py
# Funktionalität entspricht deinem mtcnn-code (Rotation A/B wählen + Paper-Crop).
# pip install facenet-pytorch pillow torch torchvision opencv-python

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from facenet_pytorch import MTCNN
from PIL import Image
import torch
import math
import numpy as np
import cv2


@dataclass
class FaceCropResult:
    face_index: int
    prob: Optional[float]
    crop: Image.Image
    eye_angle_before: float
    residual_angle_after: float
    used_rotation_sign: str  # "A(-eye_angle)" oder "B(+eye_angle)"


class MTCNNFaceCropper:
    """
    Single-image MTCNN pipeline:
      - detect faces + eye landmarks
      - rotate using eye line; tries both -eye_angle and +eye_angle
        and picks the one that makes eyes most horizontal (like your code)
      - paper crop using alpha from eye landmarks
      - returns crops (multiple faces possible)
    """

    def __init__(
        self,
        keep_all: bool = True,
        min_prob: float = 0.0,
        width_half: float = 1.3,
        device: Optional[str] = None,
        crop_scale: float = 1.15, # BEEINFLUSST CROPPING REGION
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
            borderMode=cv2.BORDER_REPLICATE,  # wie dein Code: keine schwarzen Lücken
        )

        pts = np.array(points, dtype=np.float32)  # (N,2)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  # (N,3)
        pts_rot = (M @ pts_h.T).T  # (N,2)

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
        """
        Input: PIL.Image (RGB oder wird intern nicht erzwungen; empfehlung: RGB)
        Output: Liste an FaceCropResult (kann leer sein)
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        boxes, probs, lms = self.mtcnn.detect(img, landmarks=True)

        if boxes is None or lms is None:
            return []

        # probs kann None sein -> wie dein Code: alles akzeptieren
        if probs is None:
            probs = [1.0] * len(lms)

        results: List[FaceCropResult] = []

        for i, (prob, lm) in enumerate(zip(probs, lms)):
            if prob is not None and float(prob) < self.min_prob:
                continue

            left_eye = tuple(lm[0])
            right_eye = tuple(lm[1])

            # wie dein Code: sicherstellen, dass left_eye links im Bild liegt
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

            # wie dein Code: nimm die Variante, die die Augenlinie am horizontalsten macht
            if res_a <= res_b:
                rot_img, rot_left, rot_right = rot_img_a, l_a, r_a
                used = "A(-eye_angle)"
                residual = res_a
            else:
                rot_img, rot_left, rot_right = rot_img_b, l_b, r_b
                used = "B(+eye_angle)"
                residual = res_b

            # Paper-Crop nach Rotation
            mx = (rot_left[0] + rot_right[0]) / 2.0
            my = (rot_left[1] + rot_right[1]) / 2.0
            alpha = math.hypot(rot_right[0] - mx, rot_right[1] - my)
            alpha = alpha * self.crop_scale # HIER MIT CROP SCALE MULTIPLIZIEREN

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

    def process_path(self, img_path: Union[str, Path]) -> List[FaceCropResult]:
        img_path = Path(img_path)
        img = Image.open(img_path)
        return self.process_pil(img)

    def save_results(
        self,
        results: List[FaceCropResult],
        out_dir: Union[str, Path],
        base_stem: str,
        suffix_ext: str = ".jpg",
    ) -> List[Path]:
        """
        Speichert Ergebnisse 1:1 wie dein Naming-Schema:
          <stem>_face{i}_p{prob}.jpg
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for r in results:
            prob_str = "pNA" if r.prob is None else f"p{r.prob:.3f}"
            out_path = out_dir / f"{base_stem}_face{r.face_index}_{prob_str}{suffix_ext}"
            r.crop.save(out_path)
            saved.append(out_path)
        return saved
