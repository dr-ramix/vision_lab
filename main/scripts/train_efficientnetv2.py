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

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
from fer.metrics.classification import compute_classification_metrics

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    project_root: Path
    images_root: Path
    results_root: Path = Path("results")

    batch_size: int = 64
    num_workers: int = 4

    num_classes: int = 6
    image_size: int = 64

    epochs: int = 30
    lr: float = 3e-4
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

# -----------------------------
# Utilities
# -----------------------------
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

# -----------------------------
# Training / Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_classes: int) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_n = 0
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
    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    res = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return avg_loss, res.metrics, res.confusion

def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, scaler: torch.cuda.amp.GradScaler, use_amp: bool, grad_clip: float) -> float:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))
    return total_loss / max(total_n, 1)

@torch.no_grad()
def save_random_preview_grid(model: nn.Module, loader, device: torch.device, idx_to_class_map: Dict[int, str], out_path: Path, n: int = 20, cols: int = 5, max_batches: int = 10, title: str = "Random Test Predictions") -> None:
    model.eval()
    rng = np.random.default_rng()
    images, titles, correct_flags = [], [], []

    for xb, yb in islice(loader, max_batches):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        xb_cpu, yb_cpu, preds_cpu = xb.cpu(), yb.cpu(), preds.cpu()
        for i in rng.permutation(xb_cpu.size(0)):
            if len(images) >= n:
                break
            images.append(xb_cpu[i].clamp(0, 1).permute(1, 2, 0).numpy())
            t_idx, p_idx = int(yb_cpu[i].item()), int(preds_cpu[i].item())
            titles.append(f"TRUE: {idx_to_class_map[t_idx]}\nPRED: {idx_to_class_map[p_idx]}")
            correct_flags.append(t_idx == p_idx)
        if len(images) >= n:
            break

    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, (img, t, ok) in enumerate(zip(images, titles, correct_flags)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("green" if ok else "red")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved preview grid to: {out_path}")

def save_best_checkpoint(out_dir: Path, epoch: int, model: nn.Module, optimizer: optim.Optimizer, best_val_f1: float, cfg: Config) -> None:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": best_val_f1,
        "class_to_idx": CLASS_TO_IDX,
        "class_order": CLASS_ORDER,
        "config": cfg.__dict__,
    }, out_dir / "best.ckpt.pt")

def export_frozen_model(out_dir: Path, model: nn.Module, image_size: int) -> None:
    model_cpu = model.to("cpu").eval()
    example = torch.randn(1, 3, image_size, image_size)
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception as e:
        print(f"[WARN] torch.jit.script failed ({e}). Using trace.")
        scripted = torch.jit.trace(model_cpu, example)
    frozen = torch.jit.freeze(scripted)
    frozen.save(str(out_dir / "model_frozen.ts"))
    torch.save(model_cpu.state_dict(), out_dir / "model_state_dict.pt")

# -----------------------------
# Main Training
# -----------------------------
def run(cfg: Config) -> None:
    if cfg.num_classes != 6:
        raise ValueError("Your dataloader enforces exactly 6 classes")

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = datetime.now().strftime("fer_efficientnetv2_%Y%m%d_%H%M%S")
    results_root = cfg.results_root.resolve()
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    (out_dir / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2))
    (out_dir / "class_order.json").write_text(json.dumps(CLASS_ORDER, indent=2))

    dls = build_dataloaders(cfg.images_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader, val_loader, test_loader = dls.train, dls.val, dls.test

    # EfficientNetV2
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, cfg.num_classes)
    model = model.to(device)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights(train_loader, cfg.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs,1), eta_min=cfg.min_lr)

    best_val_f1, best_epoch, best_state, patience_left = -1.0, -1, None, cfg.early_stop_patience
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, cfg.grad_clip)
        scheduler.step()
        val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        val_f1, val_acc = float(val_metrics.get("f1_macro", 0.0)), float(val_metrics.get("accuracy", 0.0))
        lr_now = float(optimizer.param_groups[0]["lr"])
        improved = val_f1 > best_val_f1

        print(f"Epoch {epoch}/{cfg.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

        np.save(out_dir / f"val_cm_epoch_{epoch:03d}.npy", val_cm)
        (out_dir / f"val_metrics_epoch_{epoch:03d}.json").write_text(json.dumps(val_metrics, indent=2))
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1, "best?": "YES" if improved else ""})

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = cfg.early_stop_patience
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping. Best epoch {best_epoch}, best_val_f1 {best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, cfg.num_classes)
    test_acc, test_f1 = float(test_metrics.get("accuracy",0.0)), float(test_metrics.get("f1_macro",0.0))
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.4f} | f1 {test_f1:.4f}")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(out_dir / "test_cm.npy", test_cm)

    # save preview images
    split = cfg.preview_split.lower()
    preview_loader = train_loader if split=="train" else val_loader if split=="val" else test_loader
    preview_path = out_dir / f"preview_{split}.png"
    save_random_preview_grid(model, preview_loader, device, idx_to_class(), preview_path, n=cfg.preview_n, cols=cfg.preview_cols, max_batches=cfg.preview_max_batches)

    export_frozen_model(out_dir, model, cfg.image_size)
    print(f"Saved run to {out_dir}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    cfg = Config(project_root=project_root, images_root=images_root, results_root=project_root/"results", epochs=30)
    run(cfg)
