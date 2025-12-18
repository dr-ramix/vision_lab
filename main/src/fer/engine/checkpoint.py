from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_score: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_score": best_score,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if extra:
        ckpt["extra"] = extra

    torch.save(ckpt, path)

def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
