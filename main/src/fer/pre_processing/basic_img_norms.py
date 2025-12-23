from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class BasicProcessResult:
    resized_bgr: np.ndarray                # (64,64,3) uint8
    gray: np.ndarray                       # (64,64) uint8
    normalized_gray_vis: np.ndarray        # (64,64) uint8
    normalized_rgb_vis: np.ndarray         # (64,64,3) uint8  <- NEU (stacked)


class BasicImageProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        ksize: int = 7,
        eps: float = 1e-6,
        sigma_floor: float = 10.0,
        post_blur_ksize: int = 3,
        tanh_scale: float = 2.5,
    ):
        self.target_size = target_size
        self.ksize = ksize
        self.eps = eps
        self.sigma_floor = sigma_floor
        self.post_blur_ksize = post_blur_ksize
        self.tanh_scale = tanh_scale

    @staticmethod
    def intensity_norm_paper_look(
        gray: np.ndarray,
        ksize: int = 7,
        eps: float = 1e-6,
        sigma_floor: float = 10.0,
        post_blur_ksize: int = 3,
        tanh_scale: float = 2.5,
    ) -> np.ndarray:
        g = gray.astype(np.float32)

        mu = cv2.GaussianBlur(g, (ksize, ksize), 0)
        diff = g - mu

        var = cv2.GaussianBlur(diff * diff, (ksize, ksize), 0)
        sigma = np.sqrt(var)
        sigma = np.maximum(sigma, sigma_floor)

        x = diff / (sigma + eps)

        if post_blur_ksize and post_blur_ksize >= 3:
            x = cv2.GaussianBlur(x, (post_blur_ksize, post_blur_ksize), 0)

        y = np.tanh(x / tanh_scale)

        vis = ((y + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
        return vis

    @staticmethod
    def gray_to_3ch(gray_u8: np.ndarray) -> np.ndarray:
        """(H,W) -> (H,W,3) durch 3x Stack (identische Kanäle)."""
        # Option 1: np.stack
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
        # Option 2 (gleichwertig): cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    def process_bgr(self, img_bgr: np.ndarray) -> BasicProcessResult:
        resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        norm_gray = self.intensity_norm_paper_look(
            gray,
            ksize=self.ksize,
            eps=self.eps,
            sigma_floor=self.sigma_floor,
            post_blur_ksize=self.post_blur_ksize,
            tanh_scale=self.tanh_scale,
        )

        norm_rgb = self.gray_to_3ch(norm_gray)  # <- NEU

        return BasicProcessResult(
            resized_bgr=resized,
            gray=gray,
            normalized_gray_vis=norm_gray,
            normalized_rgb_vis=norm_rgb,
        )

    def process_path(self, img_path: Union[str, Path]) -> Optional[BasicProcessResult]:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        return self.process_bgr(img)

    def save_normalized(
        self,
        result: BasicProcessResult,
        out_path: Union[str, Path],
        save_rgb: bool = True,  # <- NEU: standardmäßig 3-kanalig speichern
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if save_rgb:
            cv2.imwrite(str(out_path), result.normalized_rgb_vis)   # (H,W,3)
        else:
            cv2.imwrite(str(out_path), result.normalized_gray_vis)  # (H,W)

        return out_path
