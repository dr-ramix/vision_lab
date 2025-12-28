from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class TrainSettings:

    model: str = "resnet18"

    project_root: Optional[Path] = None
    images_root: Optional[Path] = None
    output_root: Optional[Path] = None 

    # training basics
    epochs: int = 30
    bs: int = 64
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    num_workers: int = 4
    seed: int = 42

    # methods
    optimizer: str = "AdamW"         # AdamW | SGD
    scheduler: str = "cosine"        # cosine | none
    class_weight: bool = True
    early_stop: int = 8              # 0 disables
    grad_clip: float = 1.0
    amp: bool = True                 # auto disabled on CPU
    transfer: bool = False
    freeze_backbone: bool = False
    label_smoothing: float = 0.0     # 0..0.2 recommended
    sampler: str = "none"            # none | weighted (optional later)

    # preview
    preview_n: int = 25
    preview_split: str = "test"      # train|val|test
    preview_cols: int = 5
    preview_max_batches: int = 10

    # misc
    run_tag: str = ""                # optional string label

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # convert Paths to str for json
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d
