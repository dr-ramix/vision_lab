# main/src/fer/training/runner.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad, Adamax, NAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from fer.dataset.dataloaders.build import build_loaders, LoaderBundle
from fer.metrics.classification import compute_classification_metrics
from fer.metrics.confusion import save_confusion_matrix
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
from fer.training.losses import build_criterion as losses_build_criterion
from fer.training.losses import EmoNeXtLoss

from fer.training.augment import apply_mixup, apply_cutmix, mixed_criterion

try:
    from fer.training.ema import EMA
except Exception:
    EMA = None  # type: ignore

try:
    from fer.training.schedules import WarmupThenCosine
except Exception:
    WarmupThenCosine = None  # type: ignore

# Optional engine (preferred if present)
_ENGINE_OK = False
try:
    from fer.training.engine import train_one_epoch as engine_train_one_epoch
    from fer.training.engine import evaluate as engine_evaluate
    # if you added it in engine.py, we’ll use it; otherwise we’ll fall back
    try:
        from fer.training.engine import _make_grad_scaler as engine_make_grad_scaler
    except Exception:
        engine_make_grad_scaler = None  # type: ignore
    _ENGINE_OK = True
except Exception:
    engine_train_one_epoch = None  # type: ignore
    engine_evaluate = None  # type: ignore
    engine_make_grad_scaler = None  # type: ignore
    _ENGINE_OK = False


# ============================================================
# Public API
# ============================================================
def run_training(settings) -> Path:
    _validate_settings(settings)
    _seed_everything(int(getattr(settings, "seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(settings, "amp", True) and device.type == "cuda")

    # -------------------------
    # Create run dir + write basics
    # -------------------------
    run_dir = create_run_dir(
        Path(settings.output_root),
        model=str(settings.model),
        run_tag=str(getattr(settings, "run_tag", "")),
    )
    write_config(run_dir, settings)
    write_meta(run_dir, Path(settings.project_root))

    # -------------------------
    # Data
    # -------------------------
    bundle: LoaderBundle = build_loaders(settings)
    loaders = {"train": bundle.train, "val": bundle.val, "test": bundle.test}
    class_order = list(bundle.class_order)
    class_to_idx = dict(bundle.class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_order)

    _save_mappings(run_dir, class_to_idx=class_to_idx, class_order=class_order)

    # -------------------------
    # Model
    # -------------------------
    model = make_model(
        str(settings.model),
        num_classes=num_classes,
        in_channels=int(getattr(settings, "in_channels", 3)),
        transfer=bool(getattr(settings, "transfer", False)),
    ).to(device)

    if bool(getattr(settings, "freeze_backbone", False)) and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    num_params = _count_parameters(model)
    write_json(run_dir / "metrics" / "model_info.json", {"num_parameters": int(num_params)})

    # -------------------------
    # Loss / Optim / Scheduler / EMA
    # -------------------------
    criterion, class_weights = _build_criterion(settings, loaders["train"], device, num_classes=num_classes)
    if class_weights is not None:
        # robust saving (np.ndarray / torch.Tensor / list)
        if isinstance(class_weights, np.ndarray):
            weights_out = class_weights.tolist()
        elif isinstance(class_weights, torch.Tensor):
            weights_out = class_weights.detach().cpu().tolist()
        else:
            weights_out = list(class_weights)
        write_json(run_dir / "metrics" / "class_weights.json", {"weights": weights_out})

    optimizer = _build_optimizer(settings, model)
    scheduler = _build_scheduler(settings, optimizer)
    ema = _maybe_build_ema(settings, model)

    # mixup/cutmix
    mixup_alpha = float(getattr(settings, "mixup_alpha", 0.0))
    cutmix_alpha = float(getattr(settings, "cutmix_alpha", 0.0))
    mix_prob = float(getattr(settings, "mix_prob", 0.0))

    _print_header(
        settings,
        run_dir,
        device,
        num_params,
        class_order,
        bundle_name=str(getattr(settings, "dataloader", "main")),
    )

    # -------------------------
    # Train loop
    # -------------------------
    with Timer() as t:
        history, best = _train_loop(
            settings=settings,
            model=model,
            loaders=loaders,
            class_order=class_order,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            use_amp=use_amp,
            run_dir=run_dir,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
        )

    # -------------------------
    # Load best + test
    # -------------------------
    model.load_state_dict(best["state"])
    test_model = model

    eval_with_ema = bool(getattr(settings, "eval_with_ema", True))
    if ema is not None and eval_with_ema:
        # evaluate shadow weights (EMA)
        test_model = ema.shadow.to(device)

    test_res = _evaluate_full(test_model, loaders["test"], criterion, device, class_order)

    write_json(run_dir / "metrics" / "test_metrics.json", test_res["metrics"])
    write_json(run_dir / "metrics" / "test_per_class.json", test_res["per_class"])

    save_confusion_matrix(
        cm=test_res["confusion"],
        labels=class_order,
        out_dir=str(run_dir / "metrics"),
        model_name="test",
        normalize=str(getattr(settings, "cm_normalize", "true"))
        if getattr(settings, "cm_normalize", "true") is not None
        else None,
        save_png=bool(getattr(settings, "save_cm_png", True)),
    )

    summary = {
        "best_epoch": int(best["epoch"]),
        "best_val_score": float(best["val_score"]),
        "best_metric": str(best["metric_name"]),
        "test_loss": float(test_res["loss"]),
        "test_acc": float(test_res["metrics"].get("accuracy", 0.0)),
        "test_f1_macro": float(test_res["metrics"].get("f1_macro", 0.0)),
        "train_time_sec": float(t.seconds),
        "num_params": int(num_params),
        "num_classes": int(num_classes),
        "class_order": class_order,
        "engine_used": bool(best.get("engine_used", False)),
        "ema_used": bool(ema is not None),
        "eval_with_ema": bool(eval_with_ema and ema is not None),
    }
    write_json(run_dir / "metrics" / "final_summary.json", summary)

    # -------------------------
    # Logs (history)
    # -------------------------
    _save_history(run_dir, history)

    # -------------------------
    # Previews (mean/std unnormalization)
    # -------------------------
    split = str(getattr(settings, "preview_split", "test")).lower().strip()
    preview_loader = loaders[_normalize_split(split)]

    mean, std, stats_path = _load_train_stats_for_previews(Path(getattr(settings, "images_root")))
    write_json(
        run_dir / "previews" / "preview_input_stats.json",
        {"stats_path": stats_path, "mean": mean, "std": std},
    )

    prev = save_previews(
        model=model,
        loader=preview_loader,
        device=device,
        out_dir=run_dir / "previews",
        idx_to_class=idx_to_class,
        n=settings.preview_n,
        cols=settings.preview_cols,
        max_batches=settings.preview_max_batches,
        split_name=split,
        save_grid=True,
        seed=settings.seed,
        train_mean=mean,
        train_std=std,
        save_items_json=True,
    )

    write_json(
        run_dir / "previews" / "summary.json",
        {
            "split": prev.split,
            "saved": prev.saved,
            "grid_path": str(prev.grid_path) if prev.grid_path else None,
            "files": prev.files,
            "items": prev.items,
        },
    )

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
        run_dir,
        str(settings.model),
        float(summary["best_val_score"]),
        float(summary["test_acc"]),
        best_val_metric=str(getattr(settings, "select_metric", "f1_macro")),
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

    out_root = Path(getattr(s, "output_root"))
    out_root.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _normalize_split(split: str) -> str:
    if split not in {"train", "val", "test"}:
        return "test"
    return split


def _save_mappings(run_dir: Path, *, class_to_idx: Dict[str, int], class_order: List[str]) -> None:
    write_json(run_dir / "mappings" / "class_to_idx.json", class_to_idx)
    write_json(run_dir / "mappings" / "class_order.json", {"class_order": class_order})


def _load_train_stats_for_previews(images_root: Path) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
    """
    Locate dataset_stats_train.json for preview unnormalization.

    Your current layout writes stats to:
      <images_root>/only_mtcnn_cropped/color_and_grey/dataset_stats_train.json
      <images_root>/only_mtcnn_cropped/grey/dataset_stats_train.json

    We also keep the older fallback locations.
    """
    images_root = Path(images_root)
    
    #MAIN/BUNT:
    #images_root / "only_mtcnn_cropped" / "color_and_grey" / "dataset_stats_train.json"
    #GRAY:
    #images_root / "only_mtcnn_cropped" / "grey" / "dataset_stats_train.json"
    #HIST-EQ:
    #images_root / "images_mtcnn_cropped_norm" / "dataset_stats_train.json"
    #FER2013
    #images_root / "fer2013" / "fer2013_mtcnn_cropped_norm" / "dataset_stats_train.json"
    #FER2013 No Intensity Norm
    #images_root / "fer2013" / "fer2013_mtcnn_cropped" / "dataset_stats_train.json"
    #INT NORM
    #images_root / mtcnn_cropped_int_norm / "dataset_stats_train.json"
    candidates = [
       images_root / "only_mtcnn_cropped" / "grey" / "dataset_stats_train.json"
    ]

    seen: set[str] = set()
    for stats_path in candidates:
        sp = str(stats_path.resolve()) if stats_path.exists() else str(stats_path)
        if sp in seen:
            continue
        seen.add(sp)

        if not stats_path.exists():
            continue

        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = data.get("mean", None)
            std = data.get("std", None)
            if mean is None or std is None:
                return None, None, str(stats_path)
            return [float(x) for x in mean], [float(x) for x in std], str(stats_path)
        except Exception:
            return None, None, str(stats_path)

    # not found
    return None, None, str(candidates[0])


# ============================================================
# Loss / Optim / Scheduler / EMA
# ============================================================
def _compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in train_loader:
        y = y.view(-1)
        for i in range(num_classes):
            counts[i] += (y == i).sum()
    inv = 1.0 / counts.clamp_min(1).float()
    return inv / inv.mean()


def _build_criterion(
    settings,
    train_loader,
    device: torch.device,
    *,
    num_classes: int,
) -> Tuple[nn.Module, Optional[np.ndarray]]:
    crit, weight_t = losses_build_criterion(settings, train_loader, device, num_classes=num_classes)
    if weight_t is None:
        return crit, None
    # be robust if losses.py ever returns np.ndarray/list in future
    if isinstance(weight_t, torch.Tensor):
        return crit, weight_t.detach().cpu().numpy()
    if isinstance(weight_t, np.ndarray):
        return crit, weight_t
    return crit, np.asarray(weight_t, dtype=np.float32)


def _split_model_output(out):
    """
    Supports:
      - logits
      - (logits, extra)
      - {"logits": ..., "extra": ...}
    Returns: (logits, extra_or_none)
    """
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    if isinstance(out, dict) and "logits" in out:
        return out["logits"], out.get("extra", None)
    return out, None


def _criterion_forward(criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor, extra: Optional[dict]):
    """
    Calls criterion with extra if supported (EmoNeXtLoss), otherwise plain.
    """
    if extra is not None and isinstance(criterion, EmoNeXtLoss):
        return criterion(logits, targets, extra=extra)
    return criterion(logits, targets)


def _mixed_criterion_forward(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    extra: Optional[dict],
):
    """
    MixUp/CutMix loss that still supports EmoNeXtLoss(reg).
    """
    if isinstance(criterion, EmoNeXtLoss):
        # mix CE part
        loss = lam * criterion.ce(logits, y_a) + (1.0 - lam) * criterion.ce(logits, y_b)
        # add reg if present
        if extra is not None and "reg" in extra:
            loss = loss + criterion.lam * extra["reg"]
        return loss

    # fallback for normal CE etc.
    return mixed_criterion(criterion, logits, y_a, y_b, lam)


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

    if name == "warmup_cosine":
        if WarmupThenCosine is None:
            raise ImportError("scheduler=warmup_cosine requested, but WarmupThenCosine is not available.")
        warm = int(getattr(settings, "warmup_epochs", 3))
        min_lr = float(getattr(settings, "min_lr", 1e-6))
        base_lr = float(getattr(settings, "lr"))
        return WarmupThenCosine(
            optimizer=optimizer,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_epochs=warm,
            total_epochs=epochs,
        )

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

    raise ValueError(f"Unknown scheduler '{name}'. Choose: cosine|warmup_cosine|step|exp|plateau|none")


def _maybe_build_ema(settings, model: nn.Module):
    use_ema = bool(getattr(settings, "ema", False))
    decay = float(getattr(settings, "ema_decay", 0.9999))
    if not use_ema or decay <= 0:
        return None
    if EMA is None:
        raise ImportError("ema=true requested but fer.training.ema.EMA is not available.")
    return EMA(model, decay=decay)


# ============================================================
# Training loop
# ============================================================
def _train_loop(
    *,
    settings,
    model: nn.Module,
    loaders: Dict[str, Any],
    class_order: List[str],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema,
    device: torch.device,
    use_amp: bool,
    run_dir: Path,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    epochs = int(getattr(settings, "epochs"))
    early_stop = int(getattr(settings, "early_stop", 0))
    grad_clip = float(getattr(settings, "grad_clip", 0.0))

    # choose best metric
    metric_name = str(getattr(settings, "select_metric", "f1_macro")).strip()
    if not metric_name:
        metric_name = "f1_macro"

    best_epoch = -1
    best_val_score = -1e18
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = early_stop

    history: List[Dict[str, Any]] = []

    # Create scaler ONCE and reuse (fixes warnings + is cleaner)
    scaler = None
    if _ENGINE_OK and engine_make_grad_scaler is not None:
        scaler = engine_make_grad_scaler(use_amp)
    else:
        scaler = _make_grad_scaler_robust(use_amp)

    # Engine may not support (logits, extra) outputs / criteria needing extra.
    use_engine = bool(_ENGINE_OK and engine_train_one_epoch is not None and not isinstance(criterion, EmoNeXtLoss))

    _print_epoch_header()

    for epoch in range(1, epochs + 1):
        # -------- train --------
        if use_engine:
            # engine should accept scaler if you updated it; if not, it will ignore via **kwargs
            train_loss = float(
                engine_train_one_epoch(
                    model=model,
                    loader=loaders["train"],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    use_amp=use_amp,
                    grad_clip=grad_clip,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                    mix_prob=mix_prob,
                    ema=ema,
                    scaler=scaler,  # <- important (reused)
                )
            )
        else:
            train_loss = _train_one_epoch_basic(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                grad_clip=grad_clip,
                ema=ema,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mix_prob=mix_prob,
                scaler=scaler,
            )

        # -------- validate --------
        eval_with_ema = bool(getattr(settings, "eval_with_ema", True))
        eval_model = ema.shadow.to(device) if (ema is not None and eval_with_ema) else model

        val_res = _evaluate_full(eval_model, loaders["val"], criterion, device, class_order)

        val_score = float(val_res["metrics"].get(metric_name, -1e18))
        val_f1 = float(val_res["metrics"].get("f1_macro", 0.0))
        val_acc = float(val_res["metrics"].get("accuracy", 0.0))
        val_loss = float(val_res["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_score > best_val_score

        # metrics
        write_json(run_dir / "metrics" / f"val_metrics_epoch_{epoch:03d}.json", val_res["metrics"])
        write_json(run_dir / "metrics" / f"val_per_class_epoch_{epoch:03d}.json", val_res["per_class"])

        save_confusion_matrix(
            cm=val_res["confusion"],
            labels=class_order,
            out_dir=str(run_dir / "metrics"),
            model_name=f"val_epoch_{epoch:03d}",
            normalize=str(getattr(settings, "cm_normalize", "true"))
            if getattr(settings, "cm_normalize", "true") is not None
            else None,
            save_png=bool(getattr(settings, "save_cm_png", True)),
        )

        # checkpoints
        _save_checkpoint(
            path=run_dir / "checkpoints" / "last.pt",
            settings=settings,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_f1=float(best_val_score),  # keep field name stable in ckpt
            class_order=class_order,
            class_to_idx=class_to_idx,
        )

        if improved:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = early_stop

            _save_checkpoint(
                path=run_dir / "checkpoints" / "best.pt",
                settings=settings,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_f1=float(best_val_score),
                class_order=class_order,
                class_to_idx=class_to_idx,
            )
        else:
            if early_stop > 0:
                patience_left -= 1

        # scheduler
        if scheduler is not None:
            if scheduler.__class__.__name__ == "WarmupThenCosine":
                scheduler.step()
                lr_now = float(getattr(scheduler, "lr", lr_now))
            elif isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(lr_now),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1_macro": float(val_f1),
            "val_score": float(val_score),
            "select_metric": metric_name,
            "best": "★" if improved else "",
            "patience_left": patience_left if early_stop > 0 else "",
        }
        history.append(row)
        _print_epoch_row(row)

        if early_stop > 0 and patience_left <= 0:
            print(f"\nEarly stopping. Best epoch={best_epoch}, best {metric_name}={best_val_score:.4f}\n")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return history, {
        "epoch": best_epoch,
        "val_score": best_val_score,
        "metric_name": metric_name,
        "state": best_state,
        "engine_used": bool(use_engine),
    }


def _make_grad_scaler_robust(enabled: bool):
    """
    Torch 2.9 warning & API differences:
    - torch.cuda.amp.GradScaler(...) is deprecated
    - torch.amp.GradScaler(device_type=...) may not exist in your build
    We try a few known signatures and fall back safely.
    """
    if not enabled:
        # keep API consistent: return a scaler-like object
        try:
            return torch.cuda.amp.GradScaler(enabled=False)
        except Exception:
            return None

    # Preferred in new torch: torch.amp.GradScaler("cuda", ...)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        pass

    # Some builds accept device="cuda"
    try:
        return torch.amp.GradScaler(device="cuda", enabled=True)
    except Exception:
        pass

    # Old fallback
    return torch.cuda.amp.GradScaler(enabled=True)


def _train_one_epoch_basic(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    ema,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    scaler,
) -> float:
    """
    Fallback train loop:
      - supports AMP (robust scaler)
      - supports grad clipping
      - supports EMA
      - supports MixUp/CutMix
    """
    model.train()

    if scaler is None:
        scaler = _make_grad_scaler_robust(use_amp)

    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ----- batch augmentation: mixup/cutmix -----
        mixed = False
        y_a = y_b = None
        lam = 1.0

        if mix_prob and mix_prob > 0 and (torch.rand(1, device=xb.device).item() < mix_prob):
            if cutmix_alpha > 0 and mixup_alpha > 0:
                if torch.rand(1, device=xb.device).item() < 0.5:
                    xb, y_a, y_b, lam = apply_mixup(xb, yb, mixup_alpha)
                else:
                    xb, y_a, y_b, lam = apply_cutmix(xb, yb, cutmix_alpha)
                mixed = True
            elif mixup_alpha > 0:
                xb, y_a, y_b, lam = apply_mixup(xb, yb, mixup_alpha)
                mixed = True
            elif cutmix_alpha > 0:
                xb, y_a, y_b, lam = apply_cutmix(xb, yb, cutmix_alpha)
                mixed = True

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(xb)
            logits, extra = _split_model_output(out)

            if mixed:
                if y_a is None or y_b is None:
                    raise RuntimeError("MixUp/CutMix set mixed=True but y_a/y_b is None.")
                loss = _mixed_criterion_forward(criterion, logits, y_a, y_b, lam, extra)
            else:
                loss = _criterion_forward(criterion, logits, yb, extra)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def _evaluate_full(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_order: List[str],
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb)
        logits, extra = _split_model_output(out)
        loss = _criterion_forward(criterion, logits, yb, extra)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    res = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=len(class_order),
        class_names=class_order,
    )

    return {
        "loss": float(total_loss / max(total_n, 1)),
        "metrics": dict(res.metrics),
        "per_class": dict(res.per_class),
        "confusion": np.array(res.confusion),
        "labels": list(res.labels),
    }


def _save_checkpoint(
    *,
    path: Path,
    settings,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
    class_to_idx: Dict[str, int],
    class_order: List[str],
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_name": str(getattr(settings, "model", "")),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": float(best_val_f1),
        "class_to_idx": class_to_idx,
        "class_order": class_order,
        "config": settings.to_dict() if hasattr(settings, "to_dict") else dict(vars(settings)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ============================================================
# Logs / Exports
# ============================================================
def _save_history(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """
    IMPORTANT: your artifacts.write_json(...) expects a dict-like object in some versions.
    History is a LIST, so we store it either as:
      - logs/history.json  (raw list)
      - or wrapped dict (history: [...])
    We do BOTH for max compatibility.
    """
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) raw list JSON (most convenient)
    (logs_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    # 2) wrapped (compatible with dict-only serializers)
    write_json(logs_dir / "history_wrapped.json", {"history": history})

    # csv
    cols = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_f1_macro",
        "val_score",
        "select_metric",
        "best",
        "patience_left",
    ]
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
    torch.save(model_cpu.state_dict(), exports_dir / "model_state_dict.pt")

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
def _print_header(
    settings,
    run_dir: Path,
    device: torch.device,
    num_params: int,
    class_order: List[str],
    bundle_name: str,
) -> None:
    amp = bool(getattr(settings, "amp", True) and device.type == "cuda")
    print("\n" + "=" * 90)
    print("FER Training Run")
    print("-" * 90)
    print(f"Run dir      : {run_dir}")
    print(f"Model        : {getattr(settings, 'model', '')}")
    print(f"Dataloader   : {bundle_name}")
    print(f"Device       : {device} | AMP={'ON' if amp else 'OFF'}")
    print(f"Params       : {num_params:,}")
    print(f"Data root    : {Path(getattr(settings, 'images_root')).resolve()}")
    print(f"Num classes  : {len(class_order)}")
    print(f"Classes      : {class_order}")
    print("-" * 90)

    print("Hyperparameters")
    print(f"  epochs        : {getattr(settings, 'epochs')}")
    print(f"  bs            : {getattr(settings, 'bs')}")
    print(f"  lr            : {getattr(settings, 'lr')}")
    print(f"  min_lr        : {getattr(settings, 'min_lr', 1e-6)}")
    print(f"  weight_decay  : {getattr(settings, 'weight_decay', 1e-2)}")
    print(f"  optimizer     : {getattr(settings, 'optimizer', 'adamw')}")
    print(f"  scheduler     : {getattr(settings, 'scheduler', 'cosine')}")
    print(f"  warmup_epochs : {getattr(settings, 'warmup_epochs', 0)}")
    print(f"  loss          : {getattr(settings, 'loss', 'ce')}")
    print(f"  class_weight  : {getattr(settings, 'class_weight', True)}")
    print(f"  label_smooth  : {getattr(settings, 'label_smoothing', 0.0)}")
    print(f"  grad_clip     : {getattr(settings, 'grad_clip', 0.0)}")
    print(f"  early_stop    : {getattr(settings, 'early_stop', 0)}")
    print(f"  mix_prob      : {getattr(settings, 'mix_prob', 0.0)}")
    print(f"  mixup_alpha   : {getattr(settings, 'mixup_alpha', 0.0)}")
    print(f"  cutmix_alpha  : {getattr(settings, 'cutmix_alpha', 0.0)}")
    print(f"  ema           : {getattr(settings, 'ema', False)}")
    if bool(getattr(settings, "ema", False)):
        print(f"  ema_decay     : {getattr(settings, 'ema_decay', 0.9999)}")
        print(f"  eval_with_ema : {getattr(settings, 'eval_with_ema', True)}")
    print(f"  select_metric : {getattr(settings, 'select_metric', 'f1_macro')}")
    print("=" * 90 + "\n")


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
    print("\n" + "=" * 90)
    print("Final Summary")
    print("-" * 90)
    for k in [
        "best_epoch",
        "best_val_score",
        "best_metric",
        "test_loss",
        "test_acc",
        "test_f1_macro",
        "train_time_sec",
        "num_params",
        "num_classes",
    ]:
        print(f"{k:>18}: {summary[k]}")
    print("-" * 90)
    print("Artifacts")
    print(f"  checkpoints : {run_dir / 'checkpoints'}")
    print(f"  exports     : {run_dir / 'exports'}")
    print(f"  logs        : {run_dir / 'logs'}")
    print(f"  metrics     : {run_dir / 'metrics'}")
    print(f"  mappings    : {run_dir / 'mappings'}")
    print(f"  previews    : {run_dir / 'previews'}")
    print("=" * 90 + "\n")
