# - Dieses Skript erzeugt DataLoader, die Batches liefern:
#     xb = Bilder als Tensor
#     yb = Labels als Integer-Klassenindex
#
#
# Unsere Bilder liegen auf der Festplatte als PNG/JPG vor und sind dort
# normalerweise "uint8" Pixelwerte im Bereich 0..255.
#
# Wenn wir im Dataset die Transformation "transforms.ToTensor()" benutzen:
#   - wird das Bild in einen float32 Tensor umgewandelt
#   - und automatisch durch 255 geteilt
#
# Beispiel:
#   Pixelwert 0   -> 0.0
#   Pixelwert 255 -> 1.0
#   Pixelwert 128 -> 0.502...
#
# Wichtig:
# - Dein Modell bekommt also keine uint8 (0..255), sondern float32 (0..1).
#
# ------------------------------------------------------------
#  Was genau liefern train_loader / val_loader / test_loader?
# ------------------------------------------------------------
# Jeder DataLoader liefert in einer Schleife Batches:
#
#   for xb, yb in train_loader:
#       ...
#
# Dabei gilt:
#   xb: Tensor der Bilder
#       Shape: (B, C, H, W)
#         B = batch_size (z.B. 64)
#         C = channels (bei uns 3!)
#         H,W = Höhe/Breite (bei uns 64x64)
#
#       Beispiel:
#         xb.shape == (64, 3, 64, 64)
#
#       Datentyp:
#         xb.dtype == torch.float32
#
#       Wertebereich:
#         xb.min() >= 0.0 und xb.max() <= 1.0   (typischerweise)
#
#   yb: Tensor der Labels
#       Shape: (B,)
#         Beispiel: yb.shape == (64,)
#
#       Datentyp:
#         yb.dtype == torch.int64
#
#       Inhalt:
#         yb enthält KLASSENINDIZES (keine One-Hot Vektoren!)
#         Beispiel: [0, 3, 5, 1, 1, 2, ...]
#
# ------------------------------------------------------------
# Feste Label-Zuordnung (extrem wichtig!)
# ------------------------------------------------------------
# Wir erzwingen explizit diese Zuordnung:
#
#   anger     -> 0
#   disgust   -> 1
#   fear      -> 2
#   happiness -> 3
#   sadness   -> 4
#   surprise  -> 5
#
# Das bedeutet:
# - Wenn dein Modell "class 0" predicted, heißt das "anger".
# - Confusion Matrix / Accuracy etc. muss genau diese Reihenfolge benutzen.
#
# ------------------------------------------------------------
# Augmentations (NUR train, NICHT val/test)
# ------------------------------------------------------------
# Im train_loader werden zufällig (pro Bild, pro Epoch) Augmentations angewandt:
#
# - Horizontal Flip:
#     p = 0.5
#     -> 50% Wahrscheinlichkeit pro Sample im Train-Set
#
# - Gaussian Blur (leichter Blur):
#     p = 0.15
#     kernel_size = 3
#     sigma in [0.1, 1.0]
#
# - Contrast Veränderung:
#     p = 0.30
#     ColorJitter(contrast=0.25) bedeutet:
#         contrast factor wird zufällig in [0.75, 1.25] gewählt
#     -> manchmal wird Kontrast erhöht, manchmal verringert
#
# WICHTIG:
# - Diese Augmentations werden NICHT gespeichert.
# - Sie passieren "on-the-fly" bei jedem Zugriff in __getitem__.
# - val_loader und test_loader haben KEINE Random Augmentations,
#   damit Evaluation reproduzierbar bleibt.
#
# ------------------------------------------------------------
# Was muss das Modell erwarten?
# ------------------------------------------------------------
# INPUT:
# - 3 Kanäle (C=3), weil wir normiertes Graustufenbild auf 3 Kanäle gestackt speichern.
# - Größe: 64x64
# - Wertebereich: [0,1]
#
# Also: model input shape ist (B, 3, 64, 64)
#
# OUTPUT:
# - 6 Klassen (weil 6 Emotionen)
# - logits shape: (B, 6)
#
# ------------------------------------------------------------
# Welche Loss-Funktion?
# ------------------------------------------------------------
# Standard: torch.nn.CrossEntropyLoss()
#
# CrossEntropyLoss erwartet:
# - logits: (B, 6) float
# - target: (B,) int64 mit Klassenindex (0..5)
#
# NICHT one-hot encoden!
#
# ------------------------------------------------------------
# GPU/CPU und .to(device)
# ------------------------------------------------------------
# Wenn das Modell auf GPU ist, müssen auch die Daten auf GPU:
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
#
# for xb, yb in train_loader:
#     xb = xb.to(device)
#     yb = yb.to(device)
#     logits = model(xb)
#
# Sonst gibt's einen Fehler "tensors not on same device".
# train_cnn_vanilla_fer.py
# Trains CNNVanilla on your FER dataloaders, logs metrics + confusion matrices,
# saves best checkpoint, exports frozen TorchScript model, and saves 20 preview images
# OUTSIDE the "results" folder (as a sibling folder).
# train_cnn_vanilla_fer.py
# Readable FER training script (no CLI).
# - uses your build_dataloaders + fixed CLASS_TO_IDX mapping
# - trains CNNVanilla with AMP on GPU (if available)
# - evaluates using compute_classification_metrics (macro-F1 selection)
# - saves best checkpoint + frozen TorchScript model
# - saves first 20 preview images (TRUE/PRED) OUTSIDE the "results" folder

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from main.src.fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
from main.src.fer.metrics.classification import compute_classification_metrics
from main.src.fer.models.cnn_vanilla import CNNVanilla


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

    # preview images
    preview_n: int = 20
    preview_split: str = "test"  # "train" | "val" | "test"


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

        with torch.cuda.amp.autocast(enabled=use_amp):
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
# Preview images OUTSIDE results/
# -----------------------------
@torch.no_grad()
def save_preview_images(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    idx_to_class_map: Dict[int, str],
    n: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    saved = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        preds = model(xb).argmax(dim=1)

        xb_cpu = xb.detach().cpu()
        yb_cpu = yb.detach().cpu()
        pr_cpu = preds.detach().cpu()

        for i in range(xb_cpu.size(0)):
            if saved >= n:
                return

            img_t = xb_cpu[i].clamp(0, 1)
            img_np = (img_t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            img = Image.fromarray(img_np)

            t_idx = int(yb_cpu[i].item())
            p_idx = int(pr_cpu[i].item())
            true_name = idx_to_class_map.get(t_idx, str(t_idx))
            pred_name = idx_to_class_map.get(p_idx, str(p_idx))

            text = f"TRUE: {true_name} | PRED: {pred_name}"

            draw = ImageDraw.Draw(img)
            pad = 6
            bbox = draw.textbbox((0, 0), text, font=font)
            th = bbox[3] - bbox[1]

            draw.rectangle([0, 0, img.width, th + 2 * pad], fill=(0, 0, 0))
            draw.text((pad, pad), text, fill=(255, 255, 255), font=font)

            img.save(out_dir / f"{saved:03d}_true-{true_name}_pred-{pred_name}.png")
            saved += 1


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

    # also save raw state dict
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
    run_name = datetime.now().strftime("fer_cnnvanilla_%Y%m%d_%H%M%S")
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
    model = CNNVanilla().to(device)

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

        # best checkpoint
        if val_f1 > best_val_f1:
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
    print(
        f"TEST | loss {test_loss:.4f} | "
        f"acc {float(test_metrics.get('accuracy', 0.0)):.4f} | "
        f"f1_macro {float(test_metrics.get('f1_macro', 0.0)):.4f}"
    )
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(out_dir / "test_cm.npy", test_cm)

    # preview images OUTSIDE results/
    preview_dir = results_root.parent / f"preview_{run_name}"
    split = cfg.preview_split.lower().strip()
    preview_loader = train_loader if split == "train" else val_loader if split == "val" else test_loader

    save_preview_images(
        model=model,
        loader=preview_loader,
        device=device,
        out_dir=preview_dir,
        idx_to_class_map=idx_to_class(),
        n=cfg.preview_n,
    )
    print(f"Preview images saved to: {preview_dir}")

    # frozen model export
    export_frozen_model(out_dir, model, cfg.image_size)
    print(f"Saved run to: {out_dir}")
    print(f"- best checkpoint: {out_dir / 'best.ckpt.pt'}")
    print(f"- frozen model:    {out_dir / 'model_frozen.ts'}")
    print(f"- state_dict only: {out_dir / 'model_state_dict.pt'}")


# -----------------------------
# Set your paths here and run
# -----------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    cfg = Config(
        project_root=project_root,
        images_root=images_root,

        # you can edit these any time
        results_root=project_root / "results",
        batch_size=64,
        num_workers=4,
        epochs=30,
        lr=3e-4,
        weight_decay=1e-2,
        use_class_weights=True,
        early_stop_patience=8,
        preview_n=20,
        preview_split="test",
    )

    run(cfg)