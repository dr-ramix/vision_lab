from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class TrainSettings:
    # ==================================================
    # Core identity
    # ==================================================
    model: str = "resnet18"                 # key in fer.models.registry
    run_tag: str = ""                       # optional label appended to run dir

    # ==================================================
    # Paths (filled by train.py, not by CLI typically)
    # ==================================================
    project_root: Optional[Path] = None
    images_root: Optional[Path] = None
    output_root: Optional[Path] = None

    # ==================================================
    # Data / loader
    # ==================================================
    #MAIN/BUNT: "main", GREY : "grey", HIST-EQ: "histeq", FER2013: "fer2013", FER2013 No Int-Norm: "fer2013_no_int" 
    #INT NORM: "int_norm"
    dataloader: str = "grey"                # main | legacy | ... (see fer.dataset.dataloaders.build)
    in_channels: int = 3                    # model input channels
    image_size: int = 64                    # used for exports / dummy inputs
    num_workers: int = 4
    pin_memory: bool = True                 # effective when CUDA available

    # If your dataset is normalized (zero-centered), store stats here for previews
    # Format: (C,) values in 0..1 domain before normalization.
    data_mean: Optional[Tuple[float, float, float]] = None
    data_std: Optional[Tuple[float, float, float]] = None

    # ==================================================
    # Training basics
    # ==================================================
    epochs: int = 30
    bs: int = 64
    seed: int = 42

    # ==================================================
    # Optimization
    # ==================================================
    optimizer: str = "adamw"                # adamw | adam | sgd | rmsprop | adagrad | adamax | nadam
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2

    # optimizer extra knobs (used depending on optimizer)
    momentum: float = 0.9                   # sgd/rmsprop
    nesterov: bool = True                   # sgd
    eps: float = 1e-8                       # adam/adamw/etc
    alpha: float = 0.99                     # rmsprop

    # ==================================================
    # LR schedule
    # ==================================================
    scheduler: str = "cosine"               # cosine | warmup_cosine | step | exp | plateau | none
    warmup_epochs: int = 0                  # 0 disables (used by warmup_cosine / optional)

    # step scheduler
    step_size: int = 10
    gamma: float = 0.1                      # step/exp

    # plateau scheduler
    plateau_factor: float = 0.5
    plateau_patience: int = 2

    # ==================================================
    # Regularization / training methods
    # ==================================================
    amp: bool = True                        # auto disabled on CPU in runner
    grad_clip: float = 1.0                  # 0 disables
    early_stop: int = 12                     # 0 disables early stopping

    transfer: bool = False                  # allow pretrained weights in make_model
    freeze_backbone: bool = False           # if model supports .freeze_backbone()

    class_weight: bool = True               # compute class weights from train split
    label_smoothing: float = 0.05            # 0..0.2 recommended

    # ==================================================
    # Loss selection
    # ==================================================
    loss: str = "ce"                        # ce | emonext
    emonext_lambda: float = 0.1             # hyperparameter for EmoNeXtLoss (if used)

    # ==================================================
    # Batch-level augmentation
    # ==================================================
    mix_prob: float = 0.0                   # 0 disables
    mixup_alpha: float = 0.0                # e.g. 0.8
    cutmix_alpha: float = 0.0               # e.g. 1.0

    # ==================================================
    # EMA
    # ==================================================
    ema: bool = False
    ema_decay: float = 0.9999               # used only if ema=True
    eval_with_ema: bool = True              # if EMA enabled, evaluate EMA weights

    # ==================================================
    # Evaluation / selection criterion
    # ==================================================
    # metric name from fer.metrics.classification.MetricsResult.metrics
    select_metric: str = "accuracy"         # f1_macro | accuracy | f1_weighted | balanced_accuracy ...
    select_mode: str = "max"                # max for scores, min for loss (keep as string for clarity)

    # ==================================================
    # Preview
    # ==================================================
    preview_n: int = 25
    preview_split: str = "test"             # train | val | test
    preview_cols: int = 5
    preview_max_batches: int = 10
    preview_unnormalize: bool = True        # if normalized data, undo for visualization

    # ==================================================
    # Misc / reserved
    # ==================================================
    sampler: str = "none"                   # reserved for future (not implemented)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (Paths -> str)."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d
