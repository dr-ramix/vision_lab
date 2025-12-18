from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from fer.metrics import compute_classification_metrics

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()

    y_true_all = []
    y_pred_all = []

    for batch in tqdm(loader, desc="eval", leave=False):
        # expected batch: (images, labels) or (images, labels, meta)
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true_all.append(labels.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    result = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return result.metrics, result.confusion
