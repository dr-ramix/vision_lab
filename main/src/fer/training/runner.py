from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
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
    _validate_settings(settings)
    _seed_everything(int(getattr(settings, "seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(settings, "amp", True) and device.type == "cuda")

    # -------------------------
    # Create run + write basics
    # -------------------------
    run_dir = create_run_dir(
        Path(settings.output_root),
        model=str(settings.model),
        run_tag=str(getattr(settings, "run_tag", "")),
    )
    write_config(run_dir, settings)
    write_meta(run_dir, Path(settings.project_root))
    _save_mappings(run_dir)

    # -------------------------
    # Data
    # -------------------------
    loaders = _build_loaders(
        images_root=Path(settings.images_root),
        batch_size=int(settings.bs),
        num_workers=int(getattr(settings, "num_workers", 4)),
    )

    # -------------------------
    # Model
    # -------------------------
    model = make_model(
        str(settings.model),
        num_classes=6,
        in_channels=int(getattr(settings, "in_channels", 3)),
        transfer=bool(getattr(settings, "transfer", False)),
    ).to(device)

    if bool(getattr(settings, "freeze_backbone", False)) and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    num_params = _count_parameters(model)
    write_json(run_dir / "metrics" / "model_info.json", {"num_parameters": int(num_params)})

    # -------------------------
    # Loss / Optim / Scheduler
    # -------------------------
    criterion, class_weights = _build_criterion(settings, loaders["train"], device)
    if class_weights is not None:
        write_json(run_dir / "metrics" / "class_weights.json", {"weights": class_weights.tolist()})

    optimizer = _build_optimizer(settings, model)
    scheduler = _build_scheduler(settings, optimizer)

    _print_header(settings, run_dir, device, num_params)

    # -------------------------
    # Train
    # -------------------------
    with Timer() as t:
        history, best = _train_loop(
            settings=settings,
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=use_amp,
            run_dir=run_dir,
        )

    # -------------------------
    # Load best + test
    # -------------------------
    model.load_state_dict(best["state"])
    test_res = _evaluate(model, loaders["test"], criterion, device)

    write_json(run_dir / "metrics" / "test_metrics.json", test_res["metrics"])
    np.save(run_dir / "metrics" / "test_cm.npy", test_res["confusion"])

    summary = {
        "best_epoch": int(best["epoch"]),
        "best_val_f1_macro": float(best["val_f1"]),
        "test_loss": float(test_res["loss"]),
        "test_acc": float(test_res["metrics"].get("accuracy", 0.0)),
        "test_f1_macro": float(test_res["metrics"].get("f1_macro", 0.0)),
        "train_time_sec": float(t.seconds),
        "num_params": int(num_params),
    }
    write_json(run_dir / "metrics" / "final_summary.json", summary)

    # -------------------------
    # Logs (history)
    # -------------------------
    _save_history(run_dir, history)

    # -------------------------
    # Previews
    # -------------------------
    split = str(getattr(settings, "preview_split", "test")).lower().strip()
    preview_loader = loaders[_normalize_split(split)]
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    prev = save_previews(
        model=model,
        loader=preview_loader,
        device=device,
        out_dir=run_dir / "previews",
        idx_to_class=idx_to_class,
        n=int(getattr(settings, "preview_n", 25)),
        cols=int(getattr(settings, "preview_cols", 5)),
        max_batches=int(getattr(settings, "preview_max_batches", 10)),
        split_name=split,
        save_grid=True,
    )
    write_json(run_dir / "previews" / "summary.json", {
        "split": prev.split,
        "saved": prev.saved,
        "grid_path": str(prev.grid_path) if prev.grid_path else None,
        "files": prev.files,
        "items": prev.items,
    })

    # -------------------------
    # Exports
    # -------------------------
    _export_inference_artifacts(
        run_dir=run_dir,
        model=model,
        image_size=int(getattr(settings, "image_size", 64)),
    )

    # -------------------------
    # Timing + run index
    # -------------------------
    write_timing(run_dir, train_time_sec=t.seconds)
    append_run_index(
        Path(settings.output_root),
        run_dir=run_dir,
        model=str(settings.model),
        best_val_f1=float(best["val_f1"]),
        test_acc=float(summary["test_acc"]),
    )

    _print_footer(run_dir, summary)
    return run_dir


# ============================================================
# Setup helpers
# ============================================================
def _validate_settings(s) -> None:
    required = ["project_root", "images_root", "output_root", "model", "epochs", "bs", "lr"]
    missing = [k for k in required if not hasattr(s, k) or getattr(s, k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required setting(s): {missing}")

    if int(getattr(s, "epochs")) <= 0:
        raise ValueError("epochs must be > 0")
    if int(getattr(s, "bs")) <= 0:
        raise ValueError("bs must be > 0")
    if float(getattr(s, "lr")) <= 0:
        raise ValueError("lr must be > 0")

    images_root = Path(getattr(s, "images_root"))
    if not images_root.exists():
        raise ValueError(f"images_root does not exist: {images_root}")


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loaders(images_root: Path, batch_size: int, num_workers: int) -> Dict[str, Any]:
    dls = build_dataloaders(images_root, batch_size=batch_size, num_workers=num_workers)
    return {"train": dls.train, "val": dls.val, "test": dls.test}


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _normalize_split(split: str) -> str:
    if split not in {"train", "val", "test"}:
        return "test"
    return split


def _save_mappings(run_dir: Path) -> None:
    write_json(run_dir / "mappings" / "class_to_idx.json", CLASS_TO_IDX)
    write_json(run_dir / "mappings" / "class_order.json", {"class_order": CLASS_ORDER})


# ============================================================
# Loss / Optim / Scheduler
# ============================================================
def _compute_class_weights(train_loader) -> torch.Tensor:
    counts = torch.zeros(6, dtype=torch.long)
    for _, y in train_loader:
        y = y.view(-1)
        for i in range(6):
            counts[i] += (y == i).sum()
    inv = 1.0 / counts.clamp_min(1).float()
    return inv / inv.mean()


def _build_criterion(settings, train_loader, device: torch.device) -> Tuple[nn.Module, Optional[np.ndarray]]:
    use_w = bool(getattr(settings, "class_weight", True))
    label_smoothing = float(getattr(settings, "label_smoothing", 0.0))

    if use_w:
        w = _compute_class_weights(train_loader).to(device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing), w.detach().cpu().numpy()

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing), None


def _build_optimizer(settings, model: nn.Module) -> torch.optim.Optimizer:
    name = str(getattr(settings, "optimizer", "adamw")).strip().lower()
    lr = float(getattr(settings, "lr"))
    wd = float(getattr(settings, "weight_decay", 1e-2))

    momentum = float(getattr(settings, "momentum", 0.9))
    eps = float(getattr(settings, "eps", 1e-8))
    alpha = float(getattr(settings, "alpha", 0.99))
    nesterov = bool(getattr(settings, "nesterov", True))

    opt_map = {
        "adamw": lambda: AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=eps),
        "adam": lambda: Adam(model.parameters(), lr=lr, weight_decay=wd, eps=eps),
        "sgd": lambda: SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov),
        "rmsprop": lambda: RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, alpha=alpha, eps=eps),
        "adagrad": lambda: Adagrad(model.parameters(), lr=lr, weight_decay=wd, eps=eps),
        "adamax": lambda: Adamax(model.parameters(), lr=lr, weight_decay=wd, eps=eps),
        "nadam": lambda: NAdam(model.parameters(), lr=lr, weight_decay=wd, eps=eps),
    }
    if name not in opt_map:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: {sorted(opt_map.keys())}")
    return opt_map[name]()


def _build_scheduler(settings, optimizer: torch.optim.Optimizer):
    name = str(getattr(settings, "scheduler", "cosine")).strip().lower()
    if name in {"none", "off", ""}:
        return None

    epochs = int(getattr(settings, "epochs"))

    if name == "cosine":
        min_lr = float(getattr(settings, "min_lr", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if name == "step":
        step_size = int(getattr(settings, "step_size", 10))
        gamma = float(getattr(settings, "gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "exp":
        gamma = float(getattr(settings, "gamma", 0.98))
        return ExponentialLR(optimizer, gamma=gamma)
    if name == "plateau":
        factor = float(getattr(settings, "plateau_factor", 0.5))
        patience = int(getattr(settings, "plateau_patience", 2))
        min_lr = float(getattr(settings, "min_lr", 1e-6))
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr)

    raise ValueError(f"Unknown scheduler '{name}'. Choose: cosine|step|exp|plateau|none")


# ============================================================
# Training loop (writes checkpoints!)
# ============================================================
def _train_loop(
    *,
    settings,
    model: nn.Module,
    loaders: Dict[str, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool,
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(getattr(settings, "epochs"))
    early_stop = int(getattr(settings, "early_stop", 0))
    grad_clip = float(getattr(settings, "grad_clip", 0.0))

    best_epoch = -1
    best_val_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = early_stop

    history: List[Dict[str, Any]] = []

    _print_epoch_header()

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )

        val_res = _evaluate(model, loaders["val"], criterion, device)
        val_f1 = float(val_res["metrics"].get("f1_macro", 0.0))
        val_acc = float(val_res["metrics"].get("accuracy", 0.0))
        val_loss = float(val_res["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_f1 > best_val_f1

        # Write per-epoch metrics
        write_json(run_dir / "metrics" / f"val_metrics_epoch_{epoch:03d}.json", val_res["metrics"])
        np.save(run_dir / "metrics" / f"val_cm_epoch_{epoch:03d}.npy", val_res["confusion"])

        # Save last checkpoint (every epoch)
        _save_checkpoint(
            path=run_dir / "checkpoints" / "last.pt",
            settings=settings,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_f1=best_val_f1,
        )

        # Save best checkpoint
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = early_stop

            _save_checkpoint(
                path=run_dir / "checkpoints" / "best.pt",
                settings=settings,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_f1=best_val_f1,
            )
        else:
            if early_stop > 0:
                patience_left -= 1

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": float(train_loss),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "best": "★" if improved else "",
            "patience_left": patience_left if early_stop > 0 else "",
        }
        history.append(row)
        _print_epoch_row(row)

        if early_stop > 0 and patience_left <= 0:
            print(f"\nEarly stopping. Best epoch={best_epoch}, best_val_f1_macro={best_val_f1:.4f}\n")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return history, {"epoch": best_epoch, "val_f1": best_val_f1, "state": best_state}


def _train_one_epoch(
    *,
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

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    res = compute_classification_metrics(y_true, y_pred, num_classes=6)

    return {
        "loss": float(total_loss / max(total_n, 1)),
        "metrics": dict(res.metrics),
        "confusion": np.array(res.confusion),
    }


def _save_checkpoint(
    *,
    path: Path,
    settings,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_name": str(getattr(settings, "model", "")),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": float(best_val_f1),
        "class_to_idx": CLASS_TO_IDX,
        "class_order": CLASS_ORDER,
        "config": settings.to_dict() if hasattr(settings, "to_dict") else dict(vars(settings)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ============================================================
# Logs / Exports
# ============================================================
def _save_history(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # json
    write_json(logs_dir / "history.json", history)

    # csv (simple + readable)
    cols = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_f1_macro", "best", "patience_left"]
    csv_path = logs_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in history:
            w.writerow({c: r.get(c, "") for c in cols})


def _export_inference_artifacts(run_dir: Path, model: nn.Module, image_size: int) -> None:
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu").eval()

    # 1) state_dict
    torch.save(model_cpu.state_dict(), exports_dir / "model_state_dict.pt")

    # 2) TorchScript frozen
    example = torch.randn(1, 3, image_size, image_size)
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception:
        scripted = torch.jit.trace(model_cpu, example)

    frozen = torch.jit.freeze(scripted)
    frozen.save(str(exports_dir / "model_frozen.ts"))


# ============================================================
# Printing
# ============================================================
def _print_header(settings, run_dir: Path, device: torch.device, num_params: int) -> None:
    amp = bool(getattr(settings, "amp", True) and device.type == "cuda")
    print("\n" + "=" * 78)
    print("FER Training Run")
    print("-" * 78)
    print(f"Run dir   : {run_dir}")
    print(f"Model     : {getattr(settings, 'model', '')}")
    print(f"Device    : {device} | AMP={'ON' if amp else 'OFF'}")
    print(f"Params    : {num_params:,}")
    print(f"Data root : {Path(getattr(settings, 'images_root')).resolve()}")
    print("-" * 78)
    print("Hyperparameters")
    print(f"  epochs     : {getattr(settings, 'epochs')}")
    print(f"  bs         : {getattr(settings, 'bs')}")
    print(f"  lr         : {getattr(settings, 'lr')}")
    print(f"  optimizer  : {getattr(settings, 'optimizer', 'adamw')}")
    print(f"  scheduler  : {getattr(settings, 'scheduler', 'cosine')}")
    print(f"  class_wt   : {getattr(settings, 'class_weight', True)}")
    print(f"  early_stop : {getattr(settings, 'early_stop', 0)}")
    print("=" * 78 + "\n")


def _print_epoch_header() -> None:
    hdr = (
        f"{'Epoch':>5}  {'LR':>10}  {'TrainLoss':>10}  {'ValLoss':>10}  "
        f"{'ValAcc':>8}  {'ValF1':>8}  {'Best':>4}  {'Pat':>4}"
    )
    print(hdr)
    print("-" * len(hdr))


def _print_epoch_row(r: Dict[str, Any]) -> None:
    print(
        f"{int(r['epoch']):5d}  "
        f"{float(r['lr']):10.3e}  "
        f"{float(r['train_loss']):10.4f}  "
        f"{float(r['val_loss']):10.4f}  "
        f"{float(r['val_acc']):8.4f}  "
        f"{float(r['val_f1_macro']):8.4f}  "
        f"{str(r.get('best','')):>4}  "
        f"{str(r.get('patience_left','')):>4}"
    )


def _print_footer(run_dir: Path, summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("Final Summary")
    print("-" * 78)
    for k in ["best_epoch", "best_val_f1_macro", "test_loss", "test_acc", "test_f1_macro", "train_time_sec", "num_params"]:
        print(f"{k:>18}: {summary[k]}")
    print("-" * 78)
    print("Artifacts")
    print(f"  checkpoints : {run_dir / 'checkpoints'}")
    print(f"  exports     : {run_dir / 'exports'}")
    print(f"  logs        : {run_dir / 'logs'}")
    print(f"  metrics     : {run_dir / 'metrics'}")
    print(f"  mappings    : {run_dir / 'mappings'}")
    print(f"  previews    : {run_dir / 'previews'}")
    print("=" * 78 + "\n")
