from __future__ import annotations
import sys
from pathlib import Path

from fer.config.parse_overrides import parse_overrides
from fer.config.defaults import TrainSettings
from fer.training.artifacts import get_repo_root_from_main
from fer.training.runner import run_training

def main(argv):
    overrides = parse_overrides(argv)

    # project root = .../vision_lab/main
    project_root_main = Path(__file__).resolve().parents[1]
    repo_root = project_root_main.parent

    s = TrainSettings()

    # set default paths
    s.project_root = project_root_main
    s.output_root = repo_root / "training_output"
    s.images_root = project_root_main / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

    # apply overrides by attribute name
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise ValueError(f"Unknown option '{k}'. Add it to TrainSettings in fer/config/defaults.py")
        setattr(s, k, v)

    run_dir = run_training(s)
    print(f"\nTraining finished. Run saved to:\n{run_dir}\n")

if __name__ == "__main__":
    main(sys.argv[1:])
