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

# IMPORTANT for Slurm/batch: non-interactive backend (must be set before pyplot import)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
from fer.metrics.classification import compute_classification_metrics
from fer.models.cnn_resnet18 import ResNet18FER


# -----------------------------
# Config (edit here)
# -----------------------------
@dataclass(frozen=True)
class Config:
    # paths
    project_root: Path
    images_root: Path
    results_root: Path = Path("results")  # run folder goes inside this

    # data
    batch_size: int = 64
    num_workers: int = 4

    # task
    num_classes: int = 6
    image_size: int = 64  # for TorchScript example input

    # training
    epochs: int = 30
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # options
    seed: int = 42
    use_class_weights: bool = True
    early_stop_patience: int = 8

    # preview images (saved PNG)
    preview_n: int = 20          #test on 20 random test images
    preview_cols: int = 5        #5 columns => 4 rows for 20
    preview_split: str = "test"  #random samples from test
    preview_max_batches: int = 10  #scan only first N batches (avoids list(loader))


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Class mapping helpers
# -----------------------------
def idx_to_class() -> Dict[int, str]:
    return {v: k for k, v in CLASS_TO_IDX.items()}


def compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    """
    Inverse frequency weights (normalized). Useful for imbalanced FER.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, yb in train_loader:
        yb = yb.view(-1)
        for c in range(num_classes):
            counts[c] += (yb == c).sum()
    counts = counts.clamp_min(1)
    inv = 1.0 / counts.float()
    return inv / inv.mean()


# -----------------------------
# Simple table printer (no deps)
# -----------------------------
def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def print_table(title: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        print(f"\n{title}\n(no rows)\n")
        return

    str_rows = []
    for r in rows:
        str_rows.append([_fmt(r.get(c)) for c in columns])

    widths = []
    for ci, c in enumerate(columns):
        max_cell = max(len(str_rows[ri][ci]) for ri in range(len(str_rows)))
        widths.append(max(len(c), max_cell))

    def line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def row(vals: List[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"

    print("\n" + title)
    print(line("-"))
    print(row(columns))
    print(line("="))
    for vals in str_rows:
        print(row(vals))
    print(line("-"))


def save_history_csv(out_path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    y_true_all, y_pred_all = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        preds = logits.argmax(dim=1)
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])

    res = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return avg_loss, res.metrics, res.confusion


# -----------------------------
# Train one epoch
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # New autocast API (no FutureWarning)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return total_loss / max(total_n, 1)


# -----------------------------
# Preview grid: save 20 random test predictions to PNG
# -----------------------------
@torch.no_grad()
def save_random_preview_grid(
    model: nn.Module,
    loader,
    device: torch.device,
    idx_to_class_map: Dict[int, str],
    out_path: Path,
    n: int = 20,
    cols: int = 5,
    max_batches: int = 10,
    title: str = "Random Test Predictions",
) -> None:
    """
    Sample up to n random images from the loader WITHOUT list(loader),
    and SAVE a prediction grid to out_path (Slurm-safe).
    """
    model.eval()
    rng = np.random.default_rng()

    images, titles, correct_flags = [], [], []

    # Iterate only a few batches (prevents NFS cleanup spam)
    for xb, yb in islice(loader, max_batches):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        preds = logits.argmax(dim=1)

        xb_cpu = xb.detach().cpu()
        yb_cpu = yb.detach().cpu()
        preds_cpu = preds.detach().cpu()

        for i in rng.permutation(xb_cpu.size(0)):
            if len(images) >= n:
                break

            img = xb_cpu[i].clamp(0, 1).permute(1, 2, 0).numpy()

            t_idx = int(yb_cpu[i].item())
            p_idx = int(preds_cpu[i].item())
            true_label = idx_to_class_map.get(t_idx, str(t_idx))
            pred_label = idx_to_class_map.get(p_idx, str(p_idx))

            images.append(img)
            titles.append(f"TRUE: {true_label}\nPRED: {pred_label}")
            correct_flags.append(t_idx == p_idx)

        if len(images) >= n:
            break

    if not images:
        print("No images collected for preview.")
        return

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


# -----------------------------
# Saving
# -----------------------------
def save_best_checkpoint(
    out_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "class_to_idx": CLASS_TO_IDX,
            "class_order": CLASS_ORDER,
            "config": cfg.__dict__,
        },
        out_dir / "best.ckpt.pt",
    )


def export_frozen_model(out_dir: Path, model: nn.Module, image_size: int) -> None:
    """
    TorchScript export + freeze. This is a good 'frozen' inference artifact.
    """
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
# Main training pipeline
# -----------------------------
def run(cfg: Config) -> None:
    if cfg.num_classes != 6:
        raise ValueError("Your dataloader enforces exactly 6 classes (CLASS_ORDER).")

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # create run folder inside results/
    run_name = datetime.now().strftime("fer_resnet18_%Y%m%d_%H%M%S")
    results_root = cfg.results_root.resolve()
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config + mappings
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    (out_dir / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2))
    (out_dir / "class_order.json").write_text(json.dumps(CLASS_ORDER, indent=2))

    # dataloaders
    dls = build_dataloaders(cfg.images_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader, val_loader, test_loader = dls.train, dls.val, dls.test

    # model
    model = ResNet18FER(num_classes=cfg.num_classes, in_channels=3).to(device)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights(train_loader, cfg.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        (out_dir / "class_weights.json").write_text(json.dumps(w.detach().cpu().tolist(), indent=2))
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)

    # track best by macro-F1
    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = cfg.early_stop_patience

    # history for organized tables
    history: List[Dict[str, Any]] = []

    # train epochs
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=cfg.grad_clip,
        )
        scheduler.step()

        val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        val_f1 = float(val_metrics.get("f1_macro", 0.0))
        val_acc = float(val_metrics.get("accuracy", 0.0))
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_f1 > best_val_f1

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"val f1_macro {val_f1:.4f}"
        )

        # store epoch artifacts
        np.save(out_dir / f"val_cm_epoch_{epoch:03d}.npy", val_cm)
        (out_dir / f"val_metrics_epoch_{epoch:03d}.json").write_text(json.dumps(val_metrics, indent=2))

        # add to history
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "best?": "YES" if improved else "",
            }
        )

        # best checkpoint
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = cfg.early_stop_patience

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
        else:
            patience_left -= 1

        if cfg.early_stop_patience > 0 and patience_left <= 0:
            print(f"Early stopping. Best epoch: {best_epoch}, best val f1_macro: {best_val_f1:.4f}")
            break

    # restore best weights for test + exports
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, cfg.num_classes)
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"TEST | loss {test_loss:.4f} | "
        f"acc {test_acc:.4f} | "
        f"f1_macro {test_f1:.4f}"
    )

    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(out_dir / "test_cm.npy", test_cm)

    # Organized tables
    cols = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_f1_macro", "best?"]
    print_table("Training Summary (per epoch)", history, cols)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
    }
    print_table(
        "Final Summary",
        [summary],
        ["best_epoch", "best_val_f1_macro", "test_loss", "test_acc", "test_f1_macro"],
    )

    # save tables
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_history_csv(out_dir / "history.csv", history, cols)
    (out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))

    # -----------------------------
    # SAVE preview grid (20 random test images by default)
    # -----------------------------
    split = cfg.preview_split.lower().strip()
    preview_loader = train_loader if split == "train" else val_loader if split == "val" else test_loader

    preview_dir = out_dir / "previews"
    preview_path = preview_dir / f"preview_{split}.png"

    print(f"Saving {cfg.preview_n} RANDOM preview images from split='{split}' -> {preview_path}")
    save_random_preview_grid(
        model=model,
        loader=preview_loader,
        device=device,
        idx_to_class_map=idx_to_class(),
        out_path=preview_path,
        n=cfg.preview_n,
        cols=cfg.preview_cols,
        max_batches=cfg.preview_max_batches,
        title=f"Random {split.capitalize()} Predictions",
    )

    # frozen model export
    export_frozen_model(out_dir, model, cfg.image_size)

    print(f"Saved run to: {out_dir}")
    print(f"- best checkpoint: {out_dir / 'best.ckpt.pt'}")
    print(f"- preview image:   {preview_path}")
    print(f"- frozen model:    {out_dir / 'model_frozen.ts'}")
    print(f"- state_dict only: {out_dir / 'model_state_dict.pt'}")
    print(f"- history table:   {out_dir / 'history.csv'}")


# -----------------------------
# Set your paths here and run
# -----------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    cfg = Config(
        project_root=project_root,
        images_root=images_root,
        results_root=project_root / "results",
        batch_size=64,
        num_workers=4,
        epochs=30,
        lr=3e-4,
        weight_decay=1e-2,
        use_class_weights=True,
        early_stop_patience=10,
        preview_n=20,
        preview_cols=5,
        preview_split="test",
        preview_max_batches=10,
    )

    run(cfg)
