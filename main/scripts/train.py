from __future__ import annotations

import sys
from pathlib import Path

from fer.config.parse_overrides import parse_overrides
from fer.config.defaults import TrainSettings
from fer.training.runner import run_training


# ============================================================
# Entry point
# ============================================================
def main(argv: list[str]) -> None:
    """
    Example usage:

      python train.py model=resnet18 epochs=30 bs=64 lr=3e-4
      python train.py model=cnn_vanilla optimizer=sgd lr=1e-2 momentum=0.9
      python train.py model=resnet18 transfer=true freeze_backbone=true
    """

    # --------------------------------------------------------
    # Parse CLI overrides: key=value
    # --------------------------------------------------------
    overrides = parse_overrides(argv)

    # --------------------------------------------------------
    # Resolve project paths
    # --------------------------------------------------------
    # This file lives in: vision_lab/main/train.py
    project_root_main = Path(__file__).resolve().parent.parent
    repo_root = project_root_main.parent

    # --------------------------------------------------------
    # Create default settings
    # --------------------------------------------------------
    settings = TrainSettings()

    # Required paths (always set explicitly)
    settings.project_root = project_root_main
    settings.output_root = repo_root / "training_output"
    settings.images_root = (
            project_root_main
            / "src"
            / "fer"
            / "dataset"
            / "standardized"
            / "images_mtcnn_cropped_norm"
    )

    # --------------------------------------------------------
    # Apply user overrides
    # --------------------------------------------------------
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise ValueError(
                f"Unknown option '{key}'. "
                f"Add it to TrainSettings in fer/config/defaults.py"
            )
        setattr(settings, key, value)

    # --------------------------------------------------------
    # Final sanity checks (friendly errors)
    # --------------------------------------------------------
    _validate_settings(settings)

    # --------------------------------------------------------
    # Run training
    # --------------------------------------------------------
    run_dir = run_training(settings)

    print("\n" + "=" * 72)
    print("Training finished successfully.")
    print(f"Run saved to:\n{run_dir}")
    print("=" * 72 + "\n")


# ============================================================
# Validation helpers
# ============================================================
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

    if not Path(s.images_root).exists():
        raise ValueError(f"images_root does not exist: {s.images_root}")

    Path(s.output_root).mkdir(parents=True, exist_ok=True)


# ============================================================
# Script execution
# ============================================================
if __name__ == "__main__":
    main(sys.argv[1:])
