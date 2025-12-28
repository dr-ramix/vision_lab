# main/src/fer/training/previews.py
#
# Preview utilities:
# - Save N annotated PNGs: "TRUE: <label> | PRED: <label>"
# - Save a grid PNG composed from those annotated PNGs
#
# Designed to be:
# - simple
# - Slurm-safe (matplotlib Agg)
# - decoupled: does not depend on your runner, only on (model, loader, output dir)
#
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Slurm-safe non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class PreviewResult:
    split: str
    saved: int
    out_dir: Path
    grid_path: Optional[Path]
    files: List[str]          # filenames of saved images
    items: List[Dict[str, Any]]  # per-item metadata


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
    Save annotated previews from a dataloader.

    Outputs:
      - out_dir / "000_true-<...>_pred-<...>.png" ... up to n
      - out_dir / "grid_<split_name>.png" (optional)

    Notes:
      - Samples are taken from the first `max_batches` batches and shuffled.
      - Images are assumed to be float tensors in [0,1] shaped (B,3,H,W).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, items = _save_annotated_samples(
        model=model,
        loader=loader,
        device=device,
        out_dir=out_dir,
        idx_to_class=idx_to_class,
        n=n,
        max_batches=max_batches,
    )

    grid_path: Optional[Path] = None
    if save_grid and paths:
        grid_path = out_dir / f"grid_{split_name}.png"
        _save_grid_from_pngs(
            png_paths=paths,
            out_path=grid_path,
            cols=cols,
            title=f"Preview {split_name} (n={len(paths)})",
        )

    return PreviewResult(
        split=split_name,
        saved=len(paths),
        out_dir=out_dir,
        grid_path=grid_path,
        files=[p.name for p in paths],
        items=items,
    )


@torch.no_grad()
def _save_annotated_samples(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    idx_to_class: Dict[int, str],
    n: int,
    max_batches: int,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Save up to n annotated PNGs.
    Returns (paths, items_meta).
    """
    model.eval()

    # Font is optional; PIL fallback works even if font isn't available
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    rng = np.random.default_rng()

    saved_paths: List[Path] = []
    items_meta: List[Dict[str, Any]] = []

    batches_scanned = 0
    for xb, yb in loader:
        if batches_scanned >= max_batches:
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
            if len(saved_paths) >= n:
                break

            # tensor -> uint8 RGB
            img_t = xb_cpu[int(i)].clamp(0, 1)  # (3,H,W)
            img_np = (img_t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            img = Image.fromarray(img_np)

            t_idx = int(yb_cpu[int(i)].item())
            p_idx = int(pr_cpu[int(i)].item())
            true_label = idx_to_class.get(t_idx, str(t_idx))
            pred_label = idx_to_class.get(p_idx, str(p_idx))

            correct = (t_idx == p_idx)
            text = f"TRUE: {true_label} | PRED: {pred_label}"

            _draw_label_bar(img, text=text, font=font)

            fname = f"{len(saved_paths):03d}_true-{true_label}_pred-{pred_label}.png"
            path = out_dir / fname
            img.save(path)

            saved_paths.append(path)
            items_meta.append(
                {
                    "index": len(saved_paths) - 1,
                    "true_idx": t_idx,
                    "pred_idx": p_idx,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": bool(correct),
                    "file": fname,
                }
            )

        if len(saved_paths) >= n:
            break

    return saved_paths, items_meta


def _draw_label_bar(img: Image.Image, *, text: str, font) -> None:
    """
    Draws a black bar at the top of the image and writes white text on it.
    Mutates the image in-place.
    """
    draw = ImageDraw.Draw(img)
    pad = 6

    # textbbox available in newer PIL versions; fallback if needed
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
    except Exception:
        th = 14  # fallback guess

    draw.rectangle([0, 0, img.width, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)


def _save_grid_from_pngs(*, png_paths: List[Path], out_path: Path, cols: int, title: str) -> None:
    """
    Makes a grid figure from already-annotated PNGs and saves it.
    """
    if not png_paths:
        return

    cols = max(1, int(cols))
    rows = math.ceil(len(png_paths) / cols)

    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, p in enumerate(png_paths):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(Image.open(p))
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
