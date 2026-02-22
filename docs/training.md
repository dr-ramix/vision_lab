# Training Entry Point (`train.py`)

## Usage

You can run the training script by passing configuration arguments as `key=value` pairs directly in the command line.
you find scripts in vision_lab/main/scripts

```bash
python train.py key=value key=value ...
```

### Examples

```bash
# Basic usage with a ResNet18 model
python train.py model=resnet18 epochs=30 bs=64 lr=3e-4

# Training an EmoNeXt model with custom loss and mixup
python train.py model=emonext loss=emonext mix_prob=0.5 mixup_alpha=0.8

# Training a ConvNeXt Base model with label smoothing and EMA
python train.py model=convnextferbase label_smoothing=0.1 ema=true ema_decay=0.9999
```

---

## Common CLI Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `model` | `str` | `"resnet18"` | Model architecture key (from `fer.models.registry`). Options include: `emonext`, `convnextferbase`, `coatnet`, etc. |
| `epochs` | `int` | `30` | Number of training epochs. Increase if training has not converged. |
| `bs` | `int` | `64` | Batch size (limited by your GPU memory). |
| `lr` | `float` | `3e-4` | Learning rate. This is typically the most important hyperparameter to tune. |
| `optimizer` | `str` | `"adamw"` | Optimization algorithm: `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, `adamax`, or `nadam`. |
| `scheduler` | `str` | `"cosine"` | Learning rate scheduler: `cosine`, `step`, `exp`, `plateau`, or `none`. |
| `class_weight` | `bool` | `True` | Use class-weighted loss for imbalanced datasets. |
| `label_smoothing`| `float` | `0.0` | Softens labels (values between `0.05`–`0.1` often improve generalization). |
| `loss` | `str` | `"ce"` | Loss function: `ce` or `emonext`. (Use `emonext` for EmoNeXt models). |
| `mix_prob` | `float` | `0.0` | Probability of applying MixUp / CutMix per batch. |
| `mixup_alpha` | `float` | `0.0` | MixUp strength (typical range: `0.4`–`0.8`). |
| `cutmix_alpha` | `float` | `0.0` | CutMix strength (typical range: `0.5`–`1.0`). |
| `ema` | `bool` | `False` | Enable Exponential Moving Average (EMA) of model weights. |
| `ema_decay` | `float` | `0.0` | EMA decay factor (typical value: `0.9999`). |
| `select_metric` | `str` | `"accuracy"` | Metric used to select the best checkpoint: `accuracy`, `f1_macro`, `f1_weighted`, etc. |
| `preview_split` | `str` | `"test"` | Which data split to visualize: `train`, `val`, or `test`. |

---

## Outputs

All training outputs are automatically written to a timestamped directory structured as follows:
`<output_root>/runs/<timestamp>__<model>__<...>/`

**This directory includes:**
* **Checkpoints:** `best.pt`, `last.pt`
* **Metrics:** Evaluation metrics and confusion matrices
* **Logs:** Training logs saved in CSV and JSON formats
* **Previews:** Annotated prediction images based on the `preview_split`
* **Exports:** `state_dict` and TorchScript exports