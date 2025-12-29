from __future__ import annotations

import csv
import getpass
import json
import os
import platform
import random
import string
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# Utilities
# ============================================================
def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ============================================================
# Run directory
# ============================================================
def create_run_dir(
    output_root: Path,
    *,
    model: str,
    run_tag: str = "",
) -> Path:
    """
    Create a unique run directory:
      <output_root>/runs/<timestamp>__<model>__user-<name>__<rand>/
    """
    output_root = output_root.resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user = getpass.getuser().replace(" ", "_")
    tag = f"__{run_tag}" if run_tag else ""

    run_id = f"{ts}__{model}__user-{user}{tag}__{_rand_suffix()}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Standard subfolders (keep stable!)
    for sub in ["checkpoints", "logs", "metrics", "previews", "exports", "mappings"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

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

    # Convert Paths to strings
    for k, v in list(cfg.items()):
        if isinstance(v, Path):
            cfg[k] = str(v)

    write_json(run_dir / "config.json", cfg)


# ============================================================
# Meta (environment + git)
# ============================================================
def build_meta(project_root: Path) -> Dict[str, Any]:
    """
    Environment metadata (NOT hyperparameters).
    """
    import torch

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # Git commit (best-effort)
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

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
    write_json(
        run_dir / "metrics" / "timing.json",
        {"train_time_sec": float(train_time_sec)},
    )


# ============================================================
# Global run index (optional but useful)
# ============================================================
def append_run_index(
    output_root: Path,
    *,
    run_dir: Path,
    model: str,
    best_val_f1: Optional[float],
    test_acc: Optional[float],
) -> None:
    """
    Append a row to <output_root>/runs_index.csv.
    Safe for multiple users (append-only).
    """
    index_path = output_root / "runs_index.csv"

    row = {
        "run_id": run_dir.name,
        "model": model,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        "user": getpass.getuser(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
