from __future__ import annotations
import torch
import numpy as np

def _rand_bbox(W: int, H: int, lam: float):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup(x, y, alpha: float):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]
    x = lam * x + (1.0 - lam) * x2
    return x, y, y2, lam

def apply_cutmix(x, y, alpha: float):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]

    B, C, H, W = x.shape
    x1, y1, x2b, y2b = _rand_bbox(W, H, lam)
    x[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]

    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / (W * H))
    return x, y, y2, float(lam_adj)

def mixed_criterion(criterion, logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
