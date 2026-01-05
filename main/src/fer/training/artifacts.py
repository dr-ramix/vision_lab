from __future__ import annotations

import csv
import getpass
import json
import os
import platform
import random
import re
import string
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from fer.metrics.classification import compute_classification_metrics
from fer.metrics.confusion import save_confusion_matrix, normalize_confusion


# ============================================================
# Utilities
# ============================================================
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-z0-9_\-]+", "", x)
    return x


def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def _to_serializable(x: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable structures.
    - Path -> str
    - numpy scalars -> python scalars
    - torch tensors -> list (cpu)
    - dict/list/tuple -> recurse
    - fallback -> str(x)
    """
    try:
        import numpy as np
    except Exception:
        np = None

    try:
        import torch
    except Exception:
        torch = None

    from pathlib import Path as _Path

    if x is None:
        return None

    # primitives
    if isinstance(x, (bool, int, float, str)):
        return x

    # Path
    if isinstance(x, _Path):
        return str(x)

    # numpy
    if np is not None:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()

    # torch
    if torch is not None:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()

    # dict / list / tuple
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]

    # dataclasses
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(x):
            return _to_serializable(asdict(x))
    except Exception:
        pass

    # fallback
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    """
    Write JSON safely for dicts OR lists OR any nested mix of:
      dict/list/tuple/Path/numpy/torch
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_obj = _to_serializable(obj)
    path.write_text(json.dumps(safe_obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(list(r))


def _ensure_subdirs(run_dir: Path) -> None:
    # Keep stable and predictable output layout
    for sub in ["checkpoints", "logs", "metrics", "previews", "exports", "mappings", "notes"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


# ============================================================
# Run directory
# ============================================================
def create_run_dir(output_root: Path, *, model: str, run_tag: str = "") -> Path:
    """
    Create a unique run directory:
      <output_root>/runs/<timestamp>__<model>__user-<name>__<tag>__<rand>/
    """
    output_root = output_root.resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user = getpass.getuser().replace(" ", "_")
    tag = f"__{_slugify(run_tag)}" if run_tag else ""

    run_id = f"{ts}__{_slugify(model)}__user-{user}{tag}__{_rand_suffix()}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _ensure_subdirs(run_dir)
    return run_dir


# ============================================================
# Config
# ============================================================
def write_config(run_dir: Path, settings) -> None:
    """
    Save training configuration exactly as used.
    """
    if hasattr(settings, "to_dict"):
        cfg = settings.to_dict()
    elif hasattr(settings, "__dataclass_fields__"):
        cfg = asdict(settings)
    else:
        cfg = dict(vars(settings))

    safe = {str(k): _to_serializable(v) for k, v in cfg.items()}
    write_json(run_dir / "config.json", safe)

    # human-friendly config summary
    lines = ["FER Training Config Summary", "-" * 78]
    for k in sorted(safe.keys()):
        lines.append(f"{k:>24}: {safe[k]}")
    write_text(run_dir / "notes" / "config_summary.txt", "\n".join(lines) + "\n")


# ============================================================
# Meta (environment + git)
# ============================================================
def build_meta(project_root: Path) -> Dict[str, Any]:
    """
    Environment metadata (NOT hyperparameters).
    """
    import torch

    meta = {
        "created_at": _now_iso(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        try:
            meta["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        except Exception:
            meta["cuda_device_name_0"] = None

    # Git commit + dirty flag (best-effort)
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        meta["git_commit"] = None

    try:
        import subprocess
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_dirty"] = bool(status)
    except Exception:
        meta["git_dirty"] = None

    return meta


def write_meta(run_dir: Path, project_root: Path) -> None:
    write_json(run_dir / "meta.json", build_meta(project_root))


# ============================================================
# Timing
# ============================================================
class Timer:
    """
    Simple context timer for training.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()

    @property
    def seconds(self) -> float:
        return float(self.end - self.start)


def write_timing(run_dir: Path, *, train_time_sec: float) -> None:
    write_json(run_dir / "metrics" / "timing.json", {"train_time_sec": float(train_time_sec)})


# ============================================================
# Informative dataset / model info helpers (optional but nice)
# ============================================================
def write_loader_info(
    run_dir: Path,
    *,
    images_root: Path,
    dataloader_name: str,
    class_order: Sequence[str],
    class_to_idx: Mapping[str, int],
    train_len: Optional[int] = None,
    val_len: Optional[int] = None,
    test_len: Optional[int] = None,
    input_channels: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> None:
    info = {
        "created_at": _now_iso(),
        "images_root": str(Path(images_root).resolve()),
        "dataloader": str(dataloader_name),
        "num_classes": int(len(class_order)),
        "class_order": list(class_order),
        "class_to_idx": dict(class_to_idx),
        "sizes": {"train": train_len, "val": val_len, "test": test_len},
        "input_channels": input_channels,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    write_json(run_dir / "metrics" / "loader_info.json", info)

    lines = ["FER Loader Info", "-" * 78]
    for k in ["images_root", "dataloader", "num_classes", "input_channels", "batch_size", "num_workers", "pin_memory"]:
        lines.append(f"{k:>24}: {info.get(k)}")
    lines.append("")
    lines.append("Class order:")
    for i, c in enumerate(class_order):
        lines.append(f"  {i:2d}: {c}")
    lines.append("")
    lines.append("Split sizes:")
    lines.append(f"  train: {train_len}")
    lines.append(f"  val  : {val_len}")
    lines.append(f"  test : {test_len}")
    write_text(run_dir / "notes" / "loader_info.txt", "\n".join(lines) + "\n")


def write_model_info(
    run_dir: Path,
    *,
    model_name: str,
    num_parameters: int,
    in_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    obj = {
        "created_at": _now_iso(),
        "model_name": str(model_name),
        "num_parameters": int(num_parameters),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "extra": extra or {},
    }
    write_json(run_dir / "metrics" / "model_info.json", obj)

    lines = ["FER Model Info", "-" * 78]
    lines.append(f"{'model_name':>24}: {obj['model_name']}")
    lines.append(f"{'num_parameters':>24}: {obj['num_parameters']:,}")
    lines.append(f"{'in_channels':>24}: {obj['in_channels']}")
    lines.append(f"{'num_classes':>24}: {obj['num_classes']}")
    if obj["extra"]:
        lines.append("")
        lines.append("Extra:")
        for k, v in obj["extra"].items():
            lines.append(f"  {k}: {_to_serializable(v)}")
    write_text(run_dir / "notes" / "model_info.txt", "\n".join(lines) + "\n")


# ============================================================
# Evaluation artifacts (THIS is the important part)
# ============================================================
def write_evaluation(
    run_dir: Path,
    *,
    split: str,
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    class_order: Sequence[str],
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    normalize_mode: Optional[str] = "true",
    save_cm_png: bool = True,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics via fer.metrics.classification + save confusion via fer.metrics.confusion.

    Writes (very informative):
      metrics/
        <name>_metrics[_epoch_XXX].json
        <name>_per_class[_epoch_XXX].json
        <name>_labels[_epoch_XXX].json
        <name>_cm_files[_epoch_XXX].json        (paths returned from save_confusion_matrix)
      notes/
        <name>_summary[_epoch_XXX].txt
      plus confusion.py outputs inside metrics/confusion/...

    Returns dict:
      {"loss":..., "metrics":..., "per_class":..., "confusion":...}
    """
    split = str(split).lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train|val|test, got: {split}")

    name = _slugify(prefix) if prefix else split
    suf = f"_epoch_{int(epoch):03d}" if epoch is not None else ""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Use YOUR classification module (includes per_class + labels + cm)
    res = compute_classification_metrics(
        y_true_arr,
        y_pred_arr,
        num_classes=len(class_order),
        class_names=list(class_order),
    )

    # Attach loss if provided
    metrics = dict(res.metrics)
    if loss is not None:
        metrics["loss"] = float(loss)

    metrics_dir = run_dir / "metrics"
    notes_dir = run_dir / "notes"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save overall metrics
    write_json(metrics_dir / f"{name}_metrics{suf}.json", metrics)

    # 2) Save per-class metrics (already computed by your module)
    write_json(metrics_dir / f"{name}_per_class{suf}.json", res.per_class)

    # 3) Save labels used for this eval (explicit!)
    write_json(
        metrics_dir / f"{name}_labels{suf}.json",
        {"labels": list(res.labels), "class_order": list(class_order)},
    )

    # 4) Save confusion matrix files using YOUR confusion.py
    # Put confusion outputs in a dedicated folder for clarity
    cm_out_dir = metrics_dir / "confusion"
    cm_out_dir.mkdir(parents=True, exist_ok=True)

    cm_model_name = f"{name}{suf}"
    cm_paths = save_confusion_matrix(
        cm=res.confusion,
        labels=list(res.labels),
        out_dir=str(cm_out_dir),
        model_name=cm_model_name,
        normalize=normalize_mode,    # "true" recommended
        save_png=bool(save_cm_png),
    )
    write_json(metrics_dir / f"{name}_cm_files{suf}.json", {"paths": cm_paths, "normalize": normalize_mode})

    # 5) Extra informative notes: summary + worst classes + confusion pairs
    summary_txt = _format_eval_summary(
        name=name,
        split=split,
        epoch=epoch,
        metrics=metrics,
        per_class=res.per_class,
        cm=res.confusion,
        labels=list(res.labels),
        normalize_mode=normalize_mode,
    )
    write_text(notes_dir / f"{name}_summary{suf}.txt", summary_txt)

    return {
        "loss": metrics.get("loss", None),
        "metrics": metrics,
        "per_class": res.per_class,
        "confusion": res.confusion,
        "labels": res.labels,
        "cm_paths": cm_paths,
    }


def _format_eval_summary(
    *,
    name: str,
    split: str,
    epoch: Optional[int],
    metrics: Dict[str, Any],
    per_class: Dict[str, Dict[str, float]],
    cm: np.ndarray,
    labels: list[str],
    normalize_mode: Optional[str],
) -> str:
    lines: list[str] = []

    title = f"Evaluation Summary: {name} | split={split}"
    if epoch is not None:
        title += f" | epoch={epoch}"
    lines.append(title)
    lines.append("-" * 90)

    # Key metrics first
    key_order = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
        "f1_micro",
    ]
    for k in key_order:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"{k:>22}: {v:.6f}")
            else:
                lines.append(f"{k:>22}: {_to_serializable(v)}")

    # Confusion stats
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc_from_cm = float(correct / max(total, 1))
    lines.append("")
    lines.append("Confusion stats:")
    lines.append(f"{'labels':>22}: {labels}")
    lines.append(f"{'cm_shape':>22}: {tuple(cm.shape)}")
    lines.append(f"{'total_samples':>22}: {total}")
    lines.append(f"{'correct':>22}: {correct}")
    lines.append(f"{'acc_from_cm':>22}: {acc_from_cm:.6f}")
    lines.append(f"{'normalize_mode':>22}: {normalize_mode}")

    # Worst classes by F1 (using your per_class metrics)
    items = []
    for cls, d in per_class.items():
        items.append(
            {
                "class": cls,
                "support": float(d.get("support", 0.0)),
                "precision": float(d.get("precision", 0.0)),
                "recall": float(d.get("recall", 0.0)),
                "f1": float(d.get("f1", 0.0)),
                "sensitivity": float(d.get("sensitivity", d.get("recall", 0.0))),
                "specificity": float(d.get("specificity", 0.0)),
            }
        )
    items_sorted = sorted(items, key=lambda x: (x["f1"], x["support"]))

    lines.append("")
    lines.append("Worst classes by F1 (up to 6):")
    for it in items_sorted[: min(6, len(items_sorted))]:
        lines.append(
            f"  - {it['class']:<12} "
            f"support={int(it['support']):4d} "
            f"P={it['precision']:.3f} R={it['recall']:.3f} F1={it['f1']:.3f} "
            f"Sens={it['sensitivity']:.3f} Spec={it['specificity']:.3f}"
        )

    # Most confused pairs (true->pred) excluding diagonal
    lines.append("")
    lines.append("Most confused pairs (true -> pred), top 10 by count:")
    pairs = _top_confusions(cm, labels, k=10)
    if not pairs:
        lines.append("  (none)")
    else:
        for t, p, c in pairs:
            lines.append(f"  - {t:>12} -> {p:<12}  count={c}")

    # If normalized, show strongest confusions by probability
    lines.append("")
    lines.append("Top normalized confusions (row-normalized), top 10 by rate:")
    cmn = normalize_confusion(cm, mode="true")
    pairsn = _top_confusions(cmn, labels, k=10, as_rate=True)
    if not pairsn:
        lines.append("  (none)")
    else:
        for t, p, r in pairsn:
            lines.append(f"  - {t:>12} -> {p:<12}  rate={float(r):.4f}")

    return "\n".join(lines) + "\n"


def _top_confusions(cm: np.ndarray, labels: list[str], k: int = 10, as_rate: bool = False):
    """
    Return list of (true_label, pred_label, value) sorted desc,
    excluding diagonal. Works for counts or normalized matrices.
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = float(cm[i, j])
            if v <= 0:
                continue
            out.append((labels[i], labels[j], v))
    out.sort(key=lambda x: x[2], reverse=True)
    if as_rate:
        return [(a, b, float(v)) for (a, b, v) in out[:k]]
    return [(a, b, int(round(v))) for (a, b, v) in out[:k]]


# ============================================================
# Global run index (more informative)
# ============================================================
def append_run_index(
    output_root: Path,
    run_dir: Path,
    model: str,
    best_val_score: Optional[float] = None,
    test_acc: Optional[float] = None,
    *,
    best_val_metric: str = "f1_macro",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a row to <output_root>/runs_index.csv.

    Positional usage:
        append_run_index(output_root, run_dir, model, best_val_score, test_acc)

    Keyword usage:
        append_run_index(output_root, run_dir, model,
                         best_val_score=..., test_acc=..., best_val_metric="accuracy")

    Writes a stable CSV schema (new columns can be added later safely).
    """
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index_path = output_root / "runs_index.csv"

    row: Dict[str, Any] = {
        "run_id": str(Path(run_dir).name),
        "model": str(model),
        "best_val_metric": str(best_val_metric),
        "best_val_score": (float(best_val_score) if best_val_score is not None else None),
        "test_acc": (float(test_acc) if test_acc is not None else None),
        "user": getpass.getuser(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if extra:
        # flatten one level; stringify unknown types
        for k, v in extra.items():
            if k in row:
                continue
            row[str(k)] = v

    # --- write (append-only) ---
    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)