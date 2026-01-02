# main/src/fer/training/previews.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # Slurm-safe
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PreviewResult:
    split: str
    saved: int
    out_dir: Path
    grid_path: Optional[Path]
    files: List[str]
    items: List[Dict[str, Any]]


def save_previews(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    idx_to_class: Dict[int, str],
    n: int = 25,
    cols: int = 5,
    max_batches: int = 10,
    split_name: str = "test",
    save_grid: bool = True,
) -> PreviewResult:
    """
    Saves:
      - out_dir / "000.png" ... (clean image with title above)
      - out_dir / "grid_<split>.png" (optional)
      - out_dir / "items.json" (optional metadata, returned anyway)

    The saved per-image PNGs are matplotlib figures (not drawn into pixels).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(
        model=model,
        loader=loader,
        device=device,
        idx_to_class=idx_to_class,
        n=n,
        max_batches=max_batches,
    )

    paths, items = _save_sample_figures(samples=samples, out_dir=out_dir)

    grid_path: Optional[Path] = None
    if save_grid and paths:
        grid_path = out_dir / f"grid_{split_name}.png"
        _save_grid_from_samples(
            samples=samples,
            out_path=grid_path,
            cols=cols,
            title=f"Preview {split_name} (n={len(samples)})",
        )

    return PreviewResult(
        split=split_name,
        saved=len(paths),
        out_dir=out_dir,
        grid_path=grid_path,
        files=[p.name for p in paths],
        items=items,
    )


# ============================================================
# Collect samples (no plotting here)
# ============================================================
@torch.no_grad()
def _collect_samples(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    idx_to_class: Dict[int, str],
    n: int,
    max_batches: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      {
        "img": np.ndarray (H,W,3) uint8,
        "true_label": str,
        "pred_label": str,
        "true_idx": int,
        "pred_idx": int,
        "correct": bool,
      }
    """
    model.eval()
    rng = np.random.default_rng()

    samples: List[Dict[str, Any]] = []
    batches_scanned = 0

    for xb, yb in loader:
        if batches_scanned >= max_batches or len(samples) >= n:
            break
        batches_scanned += 1

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        pred = logits.argmax(dim=1)

        xb_cpu = xb.detach().cpu()
        yb_cpu = yb.detach().cpu()
        pr_cpu = pred.detach().cpu()

        order = rng.permutation(xb_cpu.size(0))
        for i in order:
            if len(samples) >= n:
                break

            img_t = xb_cpu[int(i)].clamp(0, 1)  # (3,H,W)
            img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")

            t_idx = int(yb_cpu[int(i)].item())
            p_idx = int(pr_cpu[int(i)].item())
            true_label = idx_to_class.get(t_idx, str(t_idx))
            pred_label = idx_to_class.get(p_idx, str(p_idx))

            samples.append(
                {
                    "img": img,
                    "true_idx": t_idx,
                    "pred_idx": p_idx,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": bool(t_idx == p_idx),
                }
            )

    return samples


# ============================================================
# Save single images as proper plots (title above)
# ============================================================
def _save_sample_figures(
    *,
    samples: List[Dict[str, Any]],
    out_dir: Path,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    paths: List[Path] = []
    items: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples):
        true_label = s["true_label"]
        pred_label = s["pred_label"]
        correct = s["correct"]

        title = f"TRUE: {true_label} | PRED: {pred_label}"
        if correct:
            title = "✓ " + title
        else:
            title = "✗ " + title

        path = out_dir / f"{idx:03d}.png"
        _save_single_plot(image=s["img"], title=title, out_path=path)

        paths.append(path)
        items.append(
            {
                "index": idx,
                "true_idx": s["true_idx"],
                "pred_idx": s["pred_idx"],
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": correct,
                "file": path.name,
            }
        )

    return paths, items


def _save_single_plot(*, image: np.ndarray, title: str, out_path: Path) -> None:
    """
    Saves ONE image as a matplotlib figure with title above.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3.4, 3.6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# Save a grid plot directly from samples (no re-loading PNGs)
# ============================================================
def _save_grid_from_samples(
    *,
    samples: List[Dict[str, Any]],
    out_path: Path,
    cols: int,
    title: str,
) -> None:
    if not samples:
        return

    cols = max(1, int(cols))
    rows = math.ceil(len(samples) / cols)

    plt.figure(figsize=(cols * 3.4, rows * 3.6))

    for i, s in enumerate(samples):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(s["img"])
        ax.axis("off")

        prefix = "✓" if s["correct"] else "✗"
        ax.set_title(f"{prefix} TRUE: {s['true_label']}\nPRED: {s['pred_label']}", fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
