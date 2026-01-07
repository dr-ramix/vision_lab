from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
from fer.metrics.classification import compute_classification_metrics

from fer.models.coatnet import coatnet_tiny

@dataclass(frozen=True)
class Config:
    project_root: Path
    images_root: Path
    results_root: Path = Path("results")

    batch_size: int = 64
    num_workers: int = 4

    num_classes: int = 6
    image_size: int = 64

    pretrain_epochs: int = 14
    finetune_epochs: int = 30

    lr_pretrain: float = 3e-4
    lr_finetune: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    seed: int = 42
    use_class_weights: bool = True
    early_stop_patience: int = 8

    preview_n: int = 20
    preview_cols: int = 5
    preview_split: str = "test"
    preview_max_batches: int = 10


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def idx_to_class() -> Dict[int, str]:
    return {v: k for k, v in CLASS_TO_IDX.items()}


def compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, yb in train_loader:
        yb = yb.view(-1)
        for c in range(num_classes):
            counts[c] += (yb == c).sum()
    counts = counts.clamp_min(1)
    inv = 1.0 / counts.float()
    return inv / inv.mean()

@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_classes: int):
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true_all, y_pred_all = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

        preds = logits.argmax(dim=1)
        y_true_all.append(yb.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    res = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return avg_loss, res.metrics, res.confusion


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)

    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return total_loss / max(total_n, 1)

def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "heads" in name


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def run(cfg: Config):
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = datetime.now().strftime("fer_coatnet_%Y%m%d_%H%M%S")
    out_dir = (cfg.results_root / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    (out_dir / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2))
    (out_dir / "class_order.json").write_text(json.dumps(CLASS_ORDER, indent=2))

    dls = build_dataloaders(cfg.images_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader, val_loader, test_loader = dls.train, dls.val, dls.test

    model = coatnet_tiny(num_classes=cfg.num_classes)
    model = model.to(device)

    # Loss
    if cfg.use_class_weights:
        w = compute_class_weights(train_loader, cfg.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    
    print("🔹 Pre-training (frozen backbone)")
    freeze_backbone(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.pretrain_epochs, eta_min=cfg.min_lr)

    best_val_f1, patience = -1.0, cfg.early_stop_patience

    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, use_amp, cfg.grad_clip)
        scheduler.step()

        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        val_f1 = float(val_metrics.get("f1_macro", 0.0))

        print(f"[PRE] Epoch {epoch}/{cfg.pretrain_epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    
    print("🔹 Fine-tuning (full model)")
    unfreeze_all(model)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.finetune_epochs, eta_min=cfg.min_lr)

    best_val_f1, patience = -1.0, cfg.early_stop_patience

    for epoch in range(1, cfg.finetune_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, use_amp, cfg.grad_clip)
        scheduler.step()

        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        val_f1 = float(val_metrics.get("f1_macro", 0.0))

        print(f"[FT] Epoch {epoch}/{cfg.finetune_epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = cfg.early_stop_patience
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience -= 1
            if patience <= 0:
                break

    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, cfg.num_classes)

    print(f"TEST | loss {test_loss:.4f} | "
          f"acc {test_metrics.get('accuracy',0):.4f} | "
          f"f1 {test_metrics.get('f1_macro',0):.4f}")

    np.save(out_dir / "test_cm.npy", test_cm)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    cfg = Config(
        project_root=project_root,
        images_root=images_root,
        results_root=project_root / "results"
    )
    run(cfg)