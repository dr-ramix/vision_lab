from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- EmoNeXt custom loss skeleton ----
class EmoNeXtLoss(nn.Module):
    def __init__(self, lam: float = 0.1, label_smoothing: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.lam = float(lam)
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, extra: Optional[dict] = None) -> torch.Tensor:
        loss = self.ce(logits, targets)

        if extra is not None and "reg" in extra:
            loss = loss + self.lam * extra["reg"]

        return loss


def compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in train_loader:
        y = y.view(-1)
        for i in range(num_classes):
            counts[i] += (y == i).sum()
    inv = 1.0 / counts.clamp_min(1).float()
    return inv / inv.mean()


def build_criterion(settings, train_loader, device: torch.device, num_classes: int) -> Tuple[nn.Module, Optional[list]]:
    use_w = bool(getattr(settings, "class_weight", True))
    label_smoothing = float(getattr(settings, "label_smoothing", 0.0))

    w = None
    if use_w:
        w = compute_class_weights(train_loader, num_classes=num_classes).to(device)

    loss_name = str(getattr(settings, "loss", "ce")).lower().strip()
    if loss_name in {"emonext", "emonextloss"}:
        lam = float(getattr(settings, "emonext_lambda", 0.1))
        crit = EmoNeXtLoss(lam=lam, label_smoothing=label_smoothing, weight=w)
    else:
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    weights_list = w.detach().cpu().tolist() if w is not None else None
    return crit, weights_list
