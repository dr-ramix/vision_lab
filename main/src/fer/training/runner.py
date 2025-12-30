from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad, Adamax, NAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from fer.dataset.dataloaders.dataloader import (
    build_dataloaders,
    CLASS_ORDER,
    CLASS_TO_IDX,
)
from fer.metrics.classification import compute_classification_metrics
from fer.models.registry import make_model
from fer.training.previews import save_previews
from fer.training.artifacts import (
    create_run_dir,
    write_config,
    write_meta,
    write_timing,
    append_run_index,
    Timer,
    write_json,
)


# ============================================================
# Public API
# ============================================================
def run_training(settings) -> Path:
    """
    Main training entry point.
    """
    _validate_settings(settings)
    _seed_everything(getattr(settings, "seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(settings, "amp", True) and device.type == "cuda")

    # --------------------------------------------------------
    # Artifacts setup
    # --------------------------------------------------------
    run_dir = create_run_dir(
        Path(settings.output_root),
        model=str(settings.model),
        run_tag=str(getattr(settings, "run_tag", "")),
    )
    write_config(run_dir, settings)
    write_meta(run_dir, Path(settings.project_root))

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    loaders = _build_loaders(
        Path(settings.images_root),
        batch_size=settings.bs,
        num_workers=getattr(settings, "num_workers", 4),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = make_model(
        settings.model,
        num_classes=6,
        in_channels=getattr(settings, "in_channels", 3),
        transfer=getattr(settings, "transfer", False),
    ).to(device)

    if getattr(settings, "freeze_backbone", False) and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    num_params = _count_parameters(model)
    write_json(run_dir / "metrics" / "model_info.json", {"num_parameters": num_params})

    # --------------------------------------------------------
    # Loss / Optim / Scheduler
    # --------------------------------------------------------
    criterion, class_weights = _build_criterion(settings, loaders["train"], device)
    if class_weights is not None:
        write_json(run_dir / "metrics" / "class_weights.json", {"weights": class_weights.tolist()})

    optimizer = _build_optimizer(settings, model)
    scheduler = _build_scheduler(settings, optimizer)

    _print_header(settings, run_dir, device, num_params)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    with Timer() as timer:
        history, best = _train_loop(
            settings,
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            device,
            use_amp,
            run_dir,
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    model.load_state_dict(best["state"])
    test_res = _evaluate(model, loaders["test"], criterion, device)

    write_json(run_dir / "metrics" / "test_metrics.json", test_res["metrics"])
    np.save(run_dir / "metrics" / "test_cm.npy", test_res["confusion"])

    summary = {
        "best_epoch": best["epoch"],
        "best_val_f1_macro": best["val_f1"],
        "test_loss": test_res["loss"],
        "test_acc": test_res["metrics"].get("accuracy", 0.0),
        "test_f1_macro": test_res["metrics"].get("f1_macro", 0.0),
        "train_time_sec": timer.seconds,
        "num_params": num_params,
    }
    write_json(run_dir / "metrics" / "final_summary.json", summary)

    # --------------------------------------------------------
    # Previews
    # --------------------------------------------------------
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    save_previews(
        model=model,
        loader=loaders[getattr(settings, "preview_split", "test")],
        device=device,
        out_dir=run_dir / "previews",
        idx_to_class=idx_to_class,
        n=getattr(settings, "preview_n", 25),
        cols=getattr(settings, "preview_cols", 5),
        max_batches=getattr(settings, "preview_max_batches", 10),
        split_name=getattr(settings, "preview_split", "test"),
        save_grid=True,
    )

    # --------------------------------------------------------
    # Index + timing
    # --------------------------------------------------------
    write_timing(run_dir, train_time_sec=timer.seconds)
    append_run_index(
        Path(settings.output_root),
        run_dir=run_dir,
        model=settings.model,
        best_val_f1=best["val_f1"],
        test_acc=summary["test_acc"],
    )

    _print_footer(run_dir, summary)
    return run_dir


# ============================================================
# Helpers
# ============================================================
def _validate_settings(s) -> None:
    for k in ["project_root", "images_root", "output_root", "model", "epochs", "bs", "lr"]:
        if not hasattr(s, k):
            raise ValueError(f"Missing required setting: {k}")


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loaders(images_root: Path, batch_size: int, num_workers: int) -> Dict[str, Any]:
    dls = build_dataloaders(images_root, batch_size=batch_size, num_workers=num_workers)
    return {"train": dls.train, "val": dls.val, "test": dls.test}


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Loss / Optim / Scheduler
# ============================================================
def _build_criterion(settings, train_loader, device):
    if getattr(settings, "class_weight", True):
        w = _compute_class_weights(train_loader).to(device)
        return nn.CrossEntropyLoss(weight=w), w
    return nn.CrossEntropyLoss(), None


def _compute_class_weights(loader):
    counts = torch.zeros(6)
    for _, y in loader:
        for i in range(6):
            counts[i] += (y == i).sum()
    inv = 1.0 / counts.clamp(min=1)
    return inv / inv.mean()


def _build_optimizer(settings, model):
    name = getattr(settings, "optimizer", "AdamW").lower()
    lr = settings.lr
    wd = getattr(settings, "weight_decay", 1e-2)

    return {
        "adamw": AdamW,
        "adam": Adam,
        "sgd": SGD,
        "rmsprop": RMSprop,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "nadam": NAdam,
    }[name](model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(settings, optimizer):
    name = getattr(settings, "scheduler", "cosine")
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=settings.epochs)
    if name == "step":
        return StepLR(optimizer, step_size=10)
    if name == "exp":
        return ExponentialLR(optimizer, gamma=0.98)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max")
    return None


# ============================================================
# Training + Eval
# ============================================================
def _train_loop(settings, model, loaders, criterion, optimizer, scheduler, device, use_amp, run_dir):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"epoch": -1, "val_f1": -1.0, "state": None}
    history = []

    for epoch in range(1, settings.epochs + 1):
        train_loss = _train_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp)
        val = _evaluate(model, loaders["val"], criterion, device)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val["metrics"]["f1_macro"])
            else:
                scheduler.step()

        improved = val["metrics"]["f1_macro"] > best["val_f1"]
        if improved:
            best = {
                "epoch": epoch,
                "val_f1": val["metrics"]["f1_macro"],
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
            }

        history.append({"epoch": epoch, "train_loss": train_loss, **val["metrics"]})

        write_json(run_dir / "metrics" / f"val_metrics_epoch_{epoch:03d}.json", val["metrics"])

    write_json(run_dir / "logs" / "history.json", history)
    return history, best


def _train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total = 0.0
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        y_true.append(yb.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    res = compute_classification_metrics(y_true, y_pred, num_classes=6)

    return {
        "loss": loss_sum / len(loader.dataset),
        "metrics": dict(res.metrics),
        "confusion": res.confusion,
    }


# ============================================================
# Printing
# ============================================================
def _print_header(settings, run_dir, device, num_params):
    print("=" * 72)
    print("FER TRAINING")
    print(f"Run dir : {run_dir}")
    print(f"Model   : {settings.model}")
    print(f"Device  : {device}")
    print(f"Params  : {num_params:,}")
    print("=" * 72)


def _print_footer(run_dir, summary):
    print("=" * 72)
    print("FINAL SUMMARY")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    print("=" * 72)
