from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from fer.engine.checkpoint import save_checkpoint
from fer.engine.evaluator import evaluate

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    out_dir: str = "outputs/run_001"
    num_classes: Optional[int] = None
    save_best_on: str = "f1_macro"  # or "accuracy"

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)

def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # always save "last"
        save_checkpoint(out_dir / "last.pth", model, optimizer=optimizer, epoch=epoch, best_score=best_score,
                        extra={"train_loss": train_loss})

        if val_loader is not None:
            val_metrics, val_cm = evaluate(model, val_loader, device, num_classes=cfg.num_classes)

            # decide "best"
            score = float(val_metrics.get(cfg.save_best_on, 0.0))
            is_best = (best_score is None) or (score > best_score)

            if is_best:
                best_score = score
                best_metrics = dict(val_metrics)
                save_checkpoint(out_dir / "best.pth", model, optimizer=optimizer, epoch=epoch, best_score=best_score,
                                extra={"val_metrics": val_metrics, "val_confusion": val_cm.tolist()})

            print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} | "
                  f"val_{cfg.save_best_on}={score:.4f} | best={best_score:.4f}")
        else:
            print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f}")

    return model, best_metrics
