# main/src/fer/training/previews.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ============================================================
# Tensor -> viewable RGB numpy
# ============================================================
def _to_3c(img: torch.Tensor) -> torch.Tensor:
    """Ensure (3,H,W) for visualization (handles C=1 and C>=3)."""
    if img.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img.shape)}")

    c, _, _ = img.shape
    if c == 1:
        return img.repeat(3, 1, 1)
    if c >= 3:
        return img[:3]

    # c == 2 -> pad last channel
    pad = img[-1:].repeat(3 - c, 1, 1)
    return torch.cat([img, pad], dim=0)


def _unnormalize_mean_std(
    img: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """
    Undo (x - mean) / std  =>  x = x * std + mean
    """
    img = img.float()
    c = img.shape[0]

    if len(mean) not in (1, c) or len(std) not in (1, c):
        raise ValueError(
            f"mean/std must have length 1 or C. Got mean={len(mean)} std={len(std)} C={c}"
        )

    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)

    if mean_t.shape[0] == 1 and c != 1:
        mean_t = mean_t.repeat(c, 1, 1)
    if std_t.shape[0] == 1 and c != 1:
        std_t = std_t.repeat(c, 1, 1)

    return img * std_t + mean_t


def tensor_to_rgb_uint8(
    img_t: torch.Tensor,
    *,
    train_mean: Optional[Sequence[float]] = None,
    train_std: Optional[Sequence[float]] = None,
    clamp: Tuple[float, float] = (0.0, 1.0),
    auto_detect_norm: bool = False,  # IMPORTANT: keep OFF
) -> np.ndarray:
    """
    Convert (C,H,W) tensor to RGB uint8 image (H,W,3).

    Always unnormalizes when mean/std are provided.
    """
    img = img_t.detach().float()
    img = _to_3c(img)

    if train_mean is not None and train_std is not None:
        img = _unnormalize_mean_std(img, train_mean, train_std)

    lo, hi = float(clamp[0]), float(clamp[1])
    img = img.clamp(lo, hi)

    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return img_np


# ============================================================
# Matplotlib saving helpers
# ============================================================
def _save_single_plot(*, image: np.ndarray, title: str, out_path: Path) -> None:
    """Save ONE image as a matplotlib figure with title above."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3.6, 3.8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_grid_plot(
    *,
    samples: List[Dict[str, Any]],
    out_path: Path,
    cols: int,
    title: str,
) -> None:
    """Save a grid plot from collected samples (matplotlib)."""
    if not samples:
        return

    cols = max(1, int(cols))
    rows = math.ceil(len(samples) / cols)

    plt.figure(figsize=(cols * 3.6, rows * 3.8))

    for i, s in enumerate(samples):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(s["img"])
        ax.axis("off")

        prefix = "✓" if s["correct"] else "✗"
        ax.set_title(
            f"{prefix} TRUE: {s['true_label']}\nPRED: {s['pred_label']}",
            fontsize=9,
        )

    # ---- main title higher + reserved space ----
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# ============================================================
# Main API
# ============================================================
@torch.no_grad()
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
    seed: Optional[int] = None,
    train_mean: Optional[Sequence[float]] = None,
    train_std: Optional[Sequence[float]] = None,
    save_items_json: bool = True,
) -> PreviewResult:
    """
    Generate preview images and grids for a dataset split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    rng = np.random.default_rng(seed)

    samples: List[Dict[str, Any]] = []
    batches_scanned = 0

    # -------- collect samples --------
    for xb, yb in loader:
        if batches_scanned >= max_batches or len(samples) >= n:
            break
        batches_scanned += 1

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        pred = logits.argmax(dim=1)

        xb_cpu = xb.cpu()
        yb_cpu = yb.cpu()
        pr_cpu = pred.cpu()

        order = rng.permutation(xb_cpu.size(0))
        for i in order:
            if len(samples) >= n:
                break

            img_np = tensor_to_rgb_uint8(
                xb_cpu[int(i)],
                train_mean=train_mean,
                train_std=train_std,
                clamp=(0.0, 1.0),
            )

            t_idx = int(yb_cpu[int(i)])
            p_idx = int(pr_cpu[int(i)])

            samples.append(
                {
                    "img": img_np,
                    "true_idx": t_idx,
                    "pred_idx": p_idx,
                    "true_label": idx_to_class.get(t_idx, str(t_idx)),
                    "pred_label": idx_to_class.get(p_idx, str(p_idx)),
                    "correct": t_idx == p_idx,
                }
            )

    # -------- save individual images --------
    saved_paths: List[Path] = []
    items: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples):
        prefix = "✓" if s["correct"] else "✗"
        title = f"{prefix} TRUE: {s['true_label']} | PRED: {s['pred_label']}"

        path = out_dir / f"{idx:03d}.png"
        _save_single_plot(image=s["img"], title=title, out_path=path)

        saved_paths.append(path)
        items.append(
            {
                "index": idx,
                "true_idx": s["true_idx"],
                "pred_idx": s["pred_idx"],
                "true_label": s["true_label"],
                "pred_label": s["pred_label"],
                "correct": s["correct"],
                "file": path.name,
            }
        )

    # -------- save grid --------
    grid_path: Optional[Path] = None
    if save_grid and samples:
        grid_path = out_dir / f"grid_{split_name}.png"
        _save_grid_plot(
            samples=samples,
            out_path=grid_path,
            cols=cols,
            title=f"Preview {split_name} (n={len(samples)})",
        )

    if save_items_json:
        (out_dir / "items.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    return PreviewResult(
        split=split_name,
        saved=len(saved_paths),
        out_dir=out_dir,
        grid_path=grid_path,
        files=[p.name for p in saved_paths],
        items=items,
    )