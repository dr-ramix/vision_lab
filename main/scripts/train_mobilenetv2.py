# ============================
# MobileNetV2 Training Setup
# ============================

import torch
import torch.nn as nn
import numpy as np
import json
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

# ----------------------------
# Config setup
# ----------------------------
cfg.image_size = 64  # Inputgröße auf 64x64 gesetzt!

if cfg.num_classes != 6:
    raise ValueError("Your dataloader enforces exactly 6 classes (CLASS_ORDER).")

seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ----------------------------
# Create results folder
# ----------------------------
run_name = datetime.now().strftime("fer_mobilenetv2_%Y%m%d_%H%M%S")
results_root = cfg.results_root.resolve()
out_dir = results_root / run_name
out_dir.mkdir(parents=True, exist_ok=True)

# Save config & mappings
(out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
(out_dir / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2))
(out_dir / "class_order.json").write_text(json.dumps(CLASS_ORDER, indent=2))

# ----------------------------
# Dataloaders
# ----------------------------
dls = build_dataloaders(cfg.images_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
train_loader, val_loader, test_loader = dls.train, dls.val, dls.test

# ----------------------------
# Model: MobileNetV2
# ----------------------------
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, cfg.num_classes)

# ----------------------------
# Loss
# ----------------------------
if cfg.use_class_weights:
    w = compute_class_weights(train_loader, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    (out_dir / "class_weights.json").write_text(json.dumps(w.detach().cpu().tolist(), indent=2))
else:
    criterion = nn.CrossEntropyLoss()

# ----------------------------
# Freeze backbone (transfer learning)
# ----------------------------
for p in model.features.parameters():
    p.requires_grad = False

model = model.to(device)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(cfg.epochs, 1),
    eta_min=cfg.min_lr
)

# ----------------------------
# Training loop (frozen backbone)
# ----------------------------
best_val_f1 = -1.0
best_epoch = -1
best_state = None
patience_left = cfg.early_stop_patience

for epoch in range(1, cfg.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, cfg.grad_clip)
    scheduler.step()

    val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)

    val_f1 = float(val_metrics.get("f1_macro", 0.0))
    val_acc = float(val_metrics.get("accuracy", 0.0))

    print(f"Epoch {epoch:02d}/{cfg.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f} | f1_macro {val_f1:.4f}")

    np.save(out_dir / f"val_cm_epoch_{epoch:03d}.npy", val_cm)
    (out_dir / f"val_metrics_epoch_{epoch:03d}.json").write_text(json.dumps(val_metrics, indent=2))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_left = cfg.early_stop_patience
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
    else:
        patience_left -= 1

    if cfg.early_stop_patience > 0 and patience_left <= 0:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best f1_macro: {best_val_f1:.4f}")
        break

# ----------------------------
# Fine-tuning: unfreeze last blocks
# ----------------------------
for p in model.features[-2:].parameters():
    p.requires_grad = True

optimizer = AdamW(model.parameters(), lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)

best_val_f1 = -1.0
best_epoch = -1
best_state = None
patience_left = cfg.early_stop_patience

for epoch in range(1, cfg.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, cfg.grad_clip)
    scheduler.step()

    val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)

    val_f1 = float(val_metrics.get("f1_macro", 0.0))
    val_acc = float(val_metrics.get("accuracy", 0.0))

    print(f"[FT] Epoch {epoch:02d}/{cfg.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f} | f1_macro {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_left = cfg.early_stop_patience
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
    else:
        patience_left -= 1

    if cfg.early_stop_patience > 0 and patience_left <= 0:
        print(f"Early stopping (FT). Best epoch: {best_epoch}, best f1_macro: {best_val_f1:.4f}")
        break
