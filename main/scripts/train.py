from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict

from fer.config.parse_overrides import parse_overrides
from fer.config.defaults import TrainSettings
from fer.training.runner import run_training


# ============================================================
# Helpers
# ============================================================
def _resolve_project_paths() -> tuple[Path, Path]:
    """
    Assumes this file lives in: <repo>/main/train.py
    Returns:
      project_root_main = <repo>/main
      repo_root         = <repo>
    """
    project_root_main = Path(__file__).resolve().parent.parent
    repo_root = project_root_main.parent
    return project_root_main, repo_root


def _load_dataset_stats(images_root: Path) -> Dict[str, Any] | None:
    """
    Loads mean/std used for zero-centered normalization so previews can unnormalize.

    Expected file (from your preprocessing script):
      <dataset_root>/dataset_stats_train.json

    Where images_root is:
      <dataset_root>/images_mtcnn_cropped_norm   (example)

    In your preprocessing script:
      stats_path = out_root_npy.parent / "dataset_stats_train.json"
    out_root_npy was:
      dataset_root/images_mtcnn_cropped/npy
    so stats sits at:
      dataset_root/images_mtcnn_cropped/dataset_stats_train.json

    We'll search a few sensible locations:
      - images_root / "dataset_stats_train.json"
      - images_root.parent / "dataset_stats_train.json"
      - images_root.parent / "dataset_stats_train.json" (common)
      - images_root.parent.parent / "dataset_stats_train.json" (fallback)
    """
    #MAIN/BUNT:
    #images_root / "only_mtcnn_cropped" / "color_and_grey" / "dataset_stats_train.json"
    #GRAY:
    #images_root / "only_mtcnn_cropped" / "grey" / "dataset_stats_train.json"
    #HIST-EQ:
    #images_root / "images_mtcnn_cropped_norm" / "dataset_stats_train.json"
    #FER2013
    #images_root / "fer2013" / "fer2013_mtcnn_cropped_norm" / "dataset_stats_train.json"
    candidates = [
        images_root / "only_mtcnn_cropped" / "color_and_grey" / "dataset_stats_train.json"
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _apply_overrides(settings: TrainSettings, overrides: Dict[str, Any]) -> None:
    """
    Apply CLI overrides only for known fields.
    """
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise ValueError(
                f"Unknown option '{key}'. "
                f"Add it to TrainSettings in fer/config/defaults.py"
            )
        setattr(settings, key, value)


def _validate_settings(s: TrainSettings) -> None:
    """
    Fail early with clear error messages.
    """
    if not s.model:
        raise ValueError("model must be specified (e.g. model=resnet18)")

    if s.epochs <= 0:
        raise ValueError("epochs must be > 0")

    if s.bs <= 0:
        raise ValueError("bs (batch size) must be > 0")

    if s.lr <= 0:
        raise ValueError("lr must be > 0")

    if s.images_root is None:
        raise ValueError("images_root must be set (TrainSettings.images_root is None)")

    if not Path(s.images_root).exists():
        raise ValueError(f"images_root does not exist: {s.images_root}")

    if s.output_root is None:
        raise ValueError("output_root must be set (TrainSettings.output_root is None)")

    Path(s.output_root).mkdir(parents=True, exist_ok=True)


def _print_effective_config(settings: TrainSettings) -> None:
    """
    Small, readable config print before training starts.
    """
    d = settings.to_dict() if hasattr(settings, "to_dict") else dict(vars(settings))
    keys = [
        "model", "run_tag",
        "dataloader", "images_root", "output_root",
        "epochs", "bs", "lr", "optimizer", "scheduler",
        "amp", "grad_clip", "early_stop",
        "class_weight", "label_smoothing",
        "loss", "emonext_lambda",
        "mix_prob", "mixup_alpha", "cutmix_alpha",
        "ema", "ema_decay", "eval_with_ema",
        "select_metric",
        "preview_split", "preview_n",
    ]

    print("\n" + "=" * 78)
    print("FER Train (effective settings)")
    print("-" * 78)
    for k in keys:
        if k in d:
            print(f"{k:>18}: {d[k]}")
    print("=" * 78 + "\n")


# ============================================================
# Entry point
# ============================================================
def main(argv: list[str]) -> None:
    """
    Example usage:

      python train.py model=resnet18 epochs=30 bs=64 lr=3e-4
      python train.py model=cnn_vanilla optimizer=sgd lr=1e-2 momentum=0.9
      python train.py model=resnet18 transfer=true freeze_backbone=true

    Extra (new):
      python train.py dataloader=main loss=emonext emonext_lambda=0.1
      python train.py mix_prob=0.5 mixup_alpha=0.8
      python train.py ema=true ema_decay=0.9999 eval_with_ema=true
    """

    # --------------------------------------------------------
    # Parse CLI overrides: key=value
    # --------------------------------------------------------
    overrides = parse_overrides(argv)

    # --------------------------------------------------------
    # Resolve project paths
    # --------------------------------------------------------
    project_root_main, repo_root = _resolve_project_paths()

    # --------------------------------------------------------
    # Create default settings
    # --------------------------------------------------------
    settings = TrainSettings()

    # Required paths (always set explicitly)
    settings.project_root = project_root_main
    settings.output_root = repo_root / "training_output"

    # Default dataset path (your normalized npy dataset)
    settings.images_root = (
        project_root_main
        / "src"
        / "fer"
        / "dataset"
        / "standardized"
    )

    # --------------------------------------------------------
    # Apply user overrides
    # --------------------------------------------------------
    _apply_overrides(settings, overrides)

    # --------------------------------------------------------
    # Auto-load dataset mean/std for preview unnormalization
    # (only if user didn't override them explicitly)
    # --------------------------------------------------------
    try:
        if getattr(settings, "data_mean", None) is None or getattr(settings, "data_std", None) is None:
            stats = _load_dataset_stats(Path(settings.images_root))
            if stats is not None:
                mean = stats.get("mean", None)
                std = stats.get("std", None)
                if mean is not None and std is not None:
                    # store as tuples for dataclass friendliness
                    settings.data_mean = tuple(float(x) for x in mean)
                    settings.data_std = tuple(float(x) for x in std)
    except Exception:
        # preview will still work (just not unnormalized)
        pass

    # --------------------------------------------------------
    # Final sanity checks (friendly errors)
    # --------------------------------------------------------
    _validate_settings(settings)

    # --------------------------------------------------------
    # Print effective config (informative train.py)
    # --------------------------------------------------------
    _print_effective_config(settings)

    # --------------------------------------------------------
    # Run training
    # --------------------------------------------------------
    run_dir = run_training(settings)

    print("\n" + "=" * 72)
    print("Training finished successfully.")
    print(f"Run saved to:\n{run_dir}")
    print("=" * 72 + "\n")


# ============================================================
# Script execution
# ============================================================
if __name__ == "__main__":
    main(sys.argv[1:])
