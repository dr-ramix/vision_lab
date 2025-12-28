# main/src/fer/training/runner.py
# A clean, component-wise FER training runner.
#
# Usage (from your script):
#   from fer.config.defaults import TrainSettings
#   from fer.training.runner import run_training
#   run_dir = run_training(settings)
#
# Requirements from your codebase:
#   - fer.dataset.dataloaders.dataloader.build_dataloaders(images_root, batch_size, num_workers)
#     returns object with .train .val .test loaders
#   - fer.dataset.dataloaders.dataloader.CLASS_TO_IDX, CLASS_ORDER
#   - fer.metrics.classification.compute_classification_metrics(y_true, y_pred, num_classes)
#   - fer.models.registry.make_model(name, num_classes, in_channels=3, transfer=False)
#
# Optional:
#   - If your model supports transfer/freeze: implement model.freeze_backbone() (optional)
#
# main/src/fer/training/runner.py
# Clean, component-wise FER training runner with:
# - key=value settings passed from your script
# - standardized run folder artifacts
# - multiple optimizers
# - professional readable console output

from __future__ import annotations

import json
import math
import os
import random
import socket
import string
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Slurm-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook

from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad, Adamax, NAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau

from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX
from fer.metrics.classification import compute_classification_metrics
from fer.models.registry import make_model
from fer.training.previews import save_previews


# ============================================================
# Public API
# ============================================================
def run_training(settings) -> Path:
    """
    Runs one training session and saves everything under:
      <output_root>/runs/<run_id>/

    Required fields on settings:
      - project_root: Path
      - images_root: Path
      - output_root: Path
      - model: str
      - epochs: int
      - bs: int
      - lr: float

    Optional fields (recommended):
      - optimizer: AdamW|Adam|SGD|RMSprop|Adagrad|Adamax|NAdam
      - scheduler: cosine|step|exp|plateau|none
      - class_weight: bool
      - early_stop: int (0 disables)
      - weight_decay: float
      - min_lr: float
      - grad_clip: float
      - amp: bool
      - seed: int
      - num_workers: int
      - label_smoothing: float
      - preview_n: int
      - preview_split: train|val|test
      - preview_cols: int
      - preview_max_batches: int
      - run_tag: str
      - transfer: bool
      - freeze_backbone: bool
    """
    _validate_settings(settings)

    run_dir = _create_run_dir(
        output_root=Path(settings.output_root),
        model=str(settings.model),
        run_tag=str(getattr(settings, "run_tag", "")).strip(),
        user=os.getenv("USER") or os.getenv("USERNAME") or "unknown",
    )

    device = _get_device()
    use_amp = bool(getattr(settings, "amp", True) and device.type == "cuda")

    _seed_everything(int(getattr(settings, "seed", 42)))

    _save_run_config(run_dir, settings)
    _save_run_meta(run_dir, settings, device=device)
    _save_class_mappings(run_dir)

    loaders = _build_loaders(
        images_root=Path(settings.images_root),
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
    )

    model = _build_model(settings, device=device)
    num_params = _count_parameters(model)
    _write_json(run_dir / "metrics" / "model_info.json", {"num_parameters": int(num_params)})

    criterion, class_weights = _build_criterion(
        settings=settings,
        train_loader=loaders["train"],
        device=device,
        num_classes=6,
    )
    if class_weights is not None:
        _write_json(run_dir / "metrics" / "class_weights.json", {"weights": class_weights.tolist()})

    optimizer = _build_optimizer(settings, model)
    scheduler = _build_scheduler(settings, optimizer)

    _print_run_header(settings, run_dir, device=device, num_params=num_params)

    started = time.time()
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
    train_time_sec = time.time() - started

    _load_best_weights(model, best)

    test_res = _evaluate_split(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        num_classes=6,
    )
    _write_json(run_dir / "metrics" / "test_metrics.json", test_res["metrics"])
    np.save(run_dir / "metrics" / "test_cm.npy", test_res["confusion"])

    summary = _build_final_summary(best=best, test=test_res, train_time_sec=train_time_sec, num_params=num_params)
    _write_json(run_dir / "metrics" / "final_summary.json", summary)

    _save_history_tables(run_dir, history)

    # ---- Previews (via fer.training.previews) ----
    split = str(getattr(settings, "preview_split", "test")).lower().strip()
    preview_loader = loaders["test"] if split == "test" else loaders["val"] if split == "val" else loaders["train"]

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    preview_result = save_previews(
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

    # Save a small metadata summary so you can browse runs without opening images
    _write_json(run_dir / "previews" / "summary.json", {
        "split": preview_result.split,
        "saved": preview_result.saved,
        "grid_path": str(preview_result.grid_path) if preview_result.grid_path else None,
        "files": preview_result.files,
        "items": preview_result.items,
    })

    _export_inference_artifacts(
        run_dir=run_dir,
        model=model,
        image_size=int(getattr(settings, "image_size", 64)),
    )

    _write_json(run_dir / "metrics" / "timing.json", {"train_time_sec": float(train_time_sec)})

    _print_run_footer(run_dir=run_dir, summary=summary)

    return run_dir


# ============================================================
# Settings / setup
# ============================================================
def _validate_settings(s) -> None:
    required = ["project_root", "images_root", "output_root", "model", "epochs", "bs", "lr"]
    missing = [k for k in required if not hasattr(s, k)]
    if missing:
        raise ValueError(f"Settings missing required fields: {missing}")

    if int(getattr(s, "epochs")) <= 0:
        raise ValueError("epochs must be > 0")
    if int(getattr(s, "bs")) <= 0:
        raise ValueError("bs must be > 0")
    if float(getattr(s, "lr")) <= 0:
        raise ValueError("lr must be > 0")


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Run directories / saving
# ============================================================
def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def _create_run_dir(output_root: Path, model: str, run_tag: str, user: str) -> Path:
    output_root = output_root.resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_user = user.replace(" ", "_")
    safe_tag = f"__{run_tag}" if run_tag else ""
    run_id = f"{ts}__{model}__user-{safe_user}{safe_tag}__{_rand_suffix()}"

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    for sub in ["checkpoints", "logs", "metrics", "previews", "exports", "mappings"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir


def _settings_to_dict(settings) -> Dict[str, Any]:
    if hasattr(settings, "to_dict"):
        return settings.to_dict()
    d = asdict(settings) if hasattr(settings, "__dataclass_fields__") else dict(vars(settings))
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _save_run_config(run_dir: Path, settings) -> None:
    _write_json(run_dir / "config.json", _settings_to_dict(settings))


def _save_run_meta(run_dir: Path, settings, device: torch.device) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model": getattr(settings, "model", None),
    }
    try:
        import subprocess
        project_root = Path(settings.project_root).resolve()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root)).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

    _write_json(run_dir / "meta.json", meta)


def _save_class_mappings(run_dir: Path) -> None:
    _write_json(run_dir / "mappings" / "class_to_idx.json", CLASS_TO_IDX)
    _write_json(run_dir / "mappings" / "class_order.json", {"class_order": CLASS_ORDER})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ============================================================
# Data
# ============================================================
def _build_loaders(images_root: Path, batch_size: int, num_workers: int) -> Dict[str, Any]:
    dls = build_dataloaders(images_root, batch_size=batch_size, num_workers=num_workers)
    return {"train": dls.train, "val": dls.val, "test": dls.test}


# ============================================================
# Model
# ============================================================
def _build_model(settings, device: torch.device) -> nn.Module:
    model = make_model(
        str(getattr(settings, "model")),
        num_classes=6,
        in_channels=int(getattr(settings, "in_channels", 3)),
        transfer=bool(getattr(settings, "transfer", False)),
    ).to(device)

    if bool(getattr(settings, "freeze_backbone", False)) and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    return model


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ============================================================
# Loss / class imbalance
# ============================================================
def _compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, yb in train_loader:
        yb = yb.view(-1)
        for c in range(num_classes):
            counts[c] += (yb == c).sum()
    counts = counts.clamp_min(1)
    inv = 1.0 / counts.float()
    return inv / inv.mean()


def _build_criterion(
    settings,
    train_loader,
    device: torch.device,
    num_classes: int,
) -> Tuple[nn.Module, Optional[np.ndarray]]:
    label_smoothing = float(getattr(settings, "label_smoothing", 0.0))
    use_class_weights = bool(getattr(settings, "class_weight", True))

    if use_class_weights:
        w = _compute_class_weights(train_loader, num_classes=num_classes).to(device)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
        return crit, w.detach().cpu().numpy()

    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return crit, None


# ============================================================
# Optimizer / scheduler
# ============================================================
def _build_optimizer(settings, model: nn.Module) -> torch.optim.Optimizer:
    name = str(getattr(settings, "optimizer", "AdamW")).strip().lower()
    lr = float(getattr(settings, "lr", 3e-4))
    wd = float(getattr(settings, "weight_decay", 1e-2))

    # Optional hyperparams for some optimizers (safe defaults)
    momentum = float(getattr(settings, "momentum", 0.9))          # SGD/RMSprop
    betas = getattr(settings, "betas", None)                      # Adam-type
    eps = float(getattr(settings, "eps", 1e-8))
    alpha = float(getattr(settings, "alpha", 0.99))               # RMSprop
    nesterov = bool(getattr(settings, "nesterov", True))          # SGD

    # Parse betas if provided as string "0.9,0.999"
    betas_t = None
    if betas is not None:
        if isinstance(betas, str):
            parts = [p.strip() for p in betas.split(",")]
            if len(parts) == 2:
                betas_t = (float(parts[0]), float(parts[1]))
        elif isinstance(betas, (tuple, list)) and len(betas) == 2:
            betas_t = (float(betas[0]), float(betas[1]))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=eps, betas=betas_t or (0.9, 0.999))
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd, eps=eps, betas=betas_t or (0.9, 0.999))
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    if name == "rmsprop":
        return RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=wd)
    if name == "adagrad":
        return Adagrad(model.parameters(), lr=lr, weight_decay=wd, eps=eps)
    if name == "adamax":
        return Adamax(model.parameters(), lr=lr, weight_decay=wd, eps=eps, betas=betas_t or (0.9, 0.999))
    if name in {"nadam", "n_adam"}:
        return NAdam(model.parameters(), lr=lr, weight_decay=wd, eps=eps, betas=betas_t or (0.9, 0.999))

    raise ValueError(
        f"Unknown optimizer '{getattr(settings,'optimizer',None)}'. "
        "Use AdamW|Adam|SGD|RMSprop|Adagrad|Adamax|NAdam."
    )


def _build_scheduler(settings, optimizer: torch.optim.Optimizer):
    """
    scheduler options:
      - cosine: CosineAnnealingLR
      - step: StepLR (step_size, gamma)
      - exp: ExponentialLR (gamma)
      - plateau: ReduceLROnPlateau (patience, factor)
      - none
    """
    name = str(getattr(settings, "scheduler", "cosine")).strip().lower()
    if name in {"none", "off", ""}:
        return None

    epochs = int(getattr(settings, "epochs", 30))

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
        # plateau scheduler needs metric; we step it in the loop using val_f1 or val_loss
        factor = float(getattr(settings, "plateau_factor", 0.5))
        patience = int(getattr(settings, "plateau_patience", 2))
        min_lr = float(getattr(settings, "min_lr", 1e-6))
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr)

    raise ValueError(f"Unknown scheduler '{getattr(settings,'scheduler',None)}'. Use cosine|step|exp|plateau|none.")


# ============================================================
# Training loop + printing
# ============================================================
def _train_loop(
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

    epochs = int(getattr(settings, "epochs", 30))
    early_stop_patience = int(getattr(settings, "early_stop", 8))
    grad_clip = float(getattr(settings, "grad_clip", 1.0))

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = early_stop_patience

    history: List[Dict[str, Any]] = []

    _print_epoch_table_header()

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

        val_res = _evaluate_split(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            num_classes=6,
        )
        val_f1 = float(val_res["metrics"].get("f1_macro", 0.0))
        val_acc = float(val_res["metrics"].get("accuracy", 0.0))
        val_loss = float(val_res["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_f1 > best_val_f1

        # Save per-epoch metrics/CM
        _write_json(run_dir / "metrics" / f"val_metrics_epoch_{epoch:03d}.json", val_res["metrics"])
        np.save(run_dir / "metrics" / f"val_cm_epoch_{epoch:03d}.npy", val_res["confusion"])

        # Always save last
        _save_checkpoint(
            path=run_dir / "checkpoints" / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_f1=best_val_f1,
            settings=settings,
        )

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_left = early_stop_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            _save_checkpoint(
                path=run_dir / "checkpoints" / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_f1=best_val_f1,
                settings=settings,
            )
        else:
            if early_stop_patience > 0:
                patience_left -= 1

        # Scheduler stepping
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # plateau expects a metric; maximize val_f1_macro
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
            "patience_left": patience_left if early_stop_patience > 0 else "",
        }
        history.append(row)
        _print_epoch_row(row)

        if early_stop_patience > 0 and patience_left <= 0:
            print(f"\nEarly stopping triggered (patience={early_stop_patience}). Best epoch={best_epoch}, best_val_f1={best_val_f1:.4f}\n")
            break

    best = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": float(best_val_f1),
        "best_state": best_state,
    }
    return history, best


def _train_one_epoch(
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

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return total_loss / max(total_n, 1)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
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

        preds = logits.argmax(dim=1)
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])

    res = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)

    return {
        "loss": float(avg_loss),
        "metrics": dict(res.metrics),
        "confusion": np.array(res.confusion),
    }


# ============================================================
# Checkpoints
# ============================================================
def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
    settings,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": float(best_val_f1),
        "class_to_idx": CLASS_TO_IDX,
        "class_order": CLASS_ORDER,
        "config": _settings_to_dict(settings),
        "model_name": getattr(settings, "model", None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_best_weights(model: nn.Module, best: Dict[str, Any]) -> None:
    state = best.get("best_state")
    if state is not None:
        model.load_state_dict(state)


# ============================================================
# History tables (CSV + XLSX)
# ============================================================
def _save_history_tables(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    cols = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_f1_macro", "best", "patience_left"]

    csv_path = run_dir / "logs" / "history.csv"
    xlsx_path = run_dir / "logs" / "history.xlsx"
    json_path = run_dir / "logs" / "history.json"

    _save_history_csv(csv_path, history, cols)
    _save_history_xlsx(xlsx_path, history, cols)
    _write_json(json_path, history)


def _save_history_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_history_xlsx(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "history"

    ws.append(cols)  # header
    for r in rows:
        ws.append([r.get(c, "") for c in cols])

    wb.save(str(path))

# ============================================================
# Export inference artifacts
# ============================================================
def _export_inference_artifacts(run_dir: Path, model: nn.Module, image_size: int) -> None:
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu").eval()
    torch.save(model_cpu.state_dict(), exports_dir / "model_state_dict.pt")

    example = torch.randn(1, 3, image_size, image_size)
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception:
        scripted = torch.jit.trace(model_cpu, example)

    frozen = torch.jit.freeze(scripted)
    frozen.save(str(exports_dir / "model_frozen.ts"))


# ============================================================
# Professional printing helpers
# ============================================================
def _print_run_header(settings, run_dir: Path, device: torch.device, num_params: int) -> None:
    def g(k, default=""):
        return getattr(settings, k, default)

    print("\n" + "=" * 78)
    print("FER Training Run")
    print("-" * 78)
    print(f"Run dir     : {run_dir}")
    print(f"Model       : {g('model')}")
    print(f"Device      : {device} | AMP={'ON' if (device.type=='cuda' and g('amp',True)) else 'OFF'}")
    print(f"Params      : {num_params:,}")
    print(f"Data root   : {Path(g('images_root')).resolve()}")
    print("-" * 78)
    print("Hyperparameters")
    print(f"  epochs    : {g('epochs')}")
    print(f"  bs        : {g('bs')}")
    print(f"  lr        : {g('lr')}")
    print(f"  optimizer : {g('optimizer','AdamW')}")
    print(f"  scheduler : {g('scheduler','cosine')}")
    print(f"  weightdec : {g('weight_decay', 1e-2)}")
    print(f"  class_wt  : {g('class_weight', True)}")
    print(f"  label_sm  : {g('label_smoothing', 0.0)}")
    print(f"  earlystop : {g('early_stop', 8)}")
    print("=" * 78 + "\n")


def _print_epoch_table_header() -> None:
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


def _print_run_footer(run_dir: Path, summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("Final Summary")
    print("-" * 78)
    print(f"Run dir           : {run_dir}")
    print(f"Best epoch        : {summary['best_epoch']}")
    print(f"Best val f1_macro : {summary['best_val_f1_macro']:.4f}")
    print(f"Test loss         : {summary['test_loss']:.4f}")
    print(f"Test acc          : {summary['test_acc']:.4f}")
    print(f"Test f1_macro     : {summary['test_f1_macro']:.4f}")
    print(f"Train time (sec)  : {summary['train_time_sec']:.1f}")
    print(f"Num parameters    : {summary['num_params']:,}")
    print("-" * 78)
    print("Artifacts")
    print(f"  best checkpoint : {run_dir / 'checkpoints' / 'best.pt'}")
    print(f"  last checkpoint : {run_dir / 'checkpoints' / 'last.pt'}")
    print(f"  history.csv     : {run_dir / 'logs' / 'history.csv'}")
    print(f"  history.xlsx    : {run_dir / 'logs' / 'history.xlsx'}")
    print(f"  previews/       : {run_dir / 'previews'}")
    print(f"  torchscript     : {run_dir / 'exports' / 'model_frozen.ts'}")
    print("=" * 78 + "\n")


# ============================================================
# Final summary helpers
# ============================================================
def _build_final_summary(best: Dict[str, Any], test: Dict[str, Any], train_time_sec: float, num_params: int) -> Dict[str, Any]:
    return {
        "best_epoch": int(best.get("best_epoch", -1)),
        "best_val_f1_macro": float(best.get("best_val_f1_macro", 0.0)),
        "test_loss": float(test["loss"]),
        "test_acc": float(test["metrics"].get("accuracy", 0.0)),
        "test_f1_macro": float(test["metrics"].get("f1_macro", 0.0)),
        "train_time_sec": float(train_time_sec),
        "num_params": int(num_params),
    }


# ============================================================
# Checkpoint helpers
# ============================================================
def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
    settings,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": float(best_val_f1),
        "class_to_idx": CLASS_TO_IDX,
        "class_order": CLASS_ORDER,
        "config": _settings_to_dict(settings),
        "model_name": getattr(settings, "model", None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_best_weights(model: nn.Module, best: Dict[str, Any]) -> None:
    state = best.get("best_state")
    if state is not None:
        model.load_state_dict(state)


def _settings_to_dict(settings) -> Dict[str, Any]:
    if hasattr(settings, "to_dict"):
        return settings.to_dict()
    d = asdict(settings) if hasattr(settings, "__dataclass_fields__") else dict(vars(settings))
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d
