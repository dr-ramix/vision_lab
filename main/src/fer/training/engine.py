from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from fer.metrics.classification import compute_classification_metrics
from fer.training.augment import apply_mixup, apply_cutmix, mixed_criterion


# ---------------------------------------------------------------------
# AMP helpers (Torch-version safe)
# ---------------------------------------------------------------------
def _make_grad_scaler(use_amp: bool) -> Any:
    """
    Torch-version safe GradScaler.
    Your torch (2.9.x) supports: torch.amp.GradScaler("cuda", enabled=...)
    Older fallback: torch.cuda.amp.GradScaler(enabled=...)
    """
    if not use_amp:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_ctx(device: torch.device, use_amp: bool):
    """
    Autocast context manager. AMP is only meaningful on CUDA.
    """
    enabled = bool(use_amp and device.type == "cuda")
    # torch.amp.autocast exists in modern torch; uses device_type positional
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


# ---------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # no autocast in eval by default (safe + consistent); add if you want
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    res = compute_classification_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "metrics": dict(res.metrics),
        "per_class": dict(res.per_class),
        "confusion": np.array(res.confusion),
        "labels": list(res.labels),
    }


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_clip: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 0.0,          # probability to apply mix augmentation
    ema: Optional[Any] = None,      # EMA object with update(model)
    scaler: Optional[Any] = None,   # pass from runner to avoid recreating every epoch
    grad_accum: int = 1,
) -> float:
    model.train()

    use_amp = bool(use_amp and device.type == "cuda")
    scaler = scaler if scaler is not None else _make_grad_scaler(use_amp)

    total_loss, total_n = 0.0, 0
    grad_accum = max(int(grad_accum), 1)
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # ----- batch augmentation: mixup/cutmix -----
        use_mix = (mix_prob > 0.0) and (torch.rand(1, device=device).item() < float(mix_prob))
        mixed = False
        y_a = y_b = None
        lam = 1.0

        if use_mix:
            if cutmix_alpha > 0 and mixup_alpha > 0:
                # choose randomly
                if torch.rand(1, device=device).item() < 0.5:
                    xb, y_a, y_b, lam = apply_mixup(xb, yb, float(mixup_alpha))
                else:
                    xb, y_a, y_b, lam = apply_cutmix(xb, yb, float(cutmix_alpha))
                mixed = True
            elif mixup_alpha > 0:
                xb, y_a, y_b, lam = apply_mixup(xb, yb, float(mixup_alpha))
                mixed = True
            elif cutmix_alpha > 0:
                xb, y_a, y_b, lam = apply_cutmix(xb, yb, float(cutmix_alpha))
                mixed = True

        with _autocast_ctx(device, use_amp):
            logits = model(xb)
            if mixed:
                loss_raw = mixed_criterion(criterion, logits, y_a, y_b, float(lam))
            else:
                loss_raw = criterion(logits, yb)

        loss = loss_raw / float(grad_accum)

        # backward + step
        scaler.scale(loss).backward()

        if (step % grad_accum == 0) or (step == len(loader)):
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if ema is not None and ((step % grad_accum == 0) or (step == len(loader))):
            ema.update(model)

        bs = xb.size(0)
        total_loss += float(loss_raw.item()) * bs
        total_n += bs

    return float(total_loss / max(total_n, 1))
