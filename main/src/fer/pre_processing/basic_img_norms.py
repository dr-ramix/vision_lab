# src/fer/pre_processing/basic_processor.py
# Funktionalität entspricht deinem intensity_norm_paper_look + resize + grayscale->norm.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class BasicProcessResult:
    resized_bgr: np.ndarray          # (64,64,3) uint8
    gray: np.ndarray                 # (64,64) uint8
    normalized_gray_vis: np.ndarray  # (64,64) uint8 (sichtbar wie bei deinem Code)


class BasicImageProcessor:
    """
    Single-image basic processing:
      - resize -> gray
      - intensity_norm_paper_look (Eq.1-style + sigma_floor + optional blur + tanh compression)
      - returns arrays; optional save
    """

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
        """
        Paper Eq.(1): x' = (x - mu)/sigma  (mu gaussian-weighted, 7x7; sigma local std)
        Danach: sanfte tanh-Kompression -> sichtbar wie Fig.6, ohne übertriebenen Kontrast.
        """
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

    def process_bgr(self, img_bgr: np.ndarray) -> BasicProcessResult:
        """
        Input: BGR uint8 (cv2.imread output)
        Output: resized BGR, gray, normalized gray (uint8)
        """
        resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        norm = self.intensity_norm_paper_look(
            gray,
            ksize=self.ksize,
            eps=self.eps,
            sigma_floor=self.sigma_floor,
            post_blur_ksize=self.post_blur_ksize,
            tanh_scale=self.tanh_scale,
        )

        return BasicProcessResult(
            resized_bgr=resized,
            gray=gray,
            normalized_gray_vis=norm,
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
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result.normalized_gray_vis)
        return out_path
