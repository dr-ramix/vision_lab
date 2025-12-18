# acc, precision, recall, f1, sensitivity, etc.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

@dataclass
class MetricsResult:
    metrics: Dict[str, float]
    confusion: np.ndarray

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> MetricsResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # macro + weighted are useful for imbalanced FER datasets
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # sensitivity in multi-class is essentially recall per class; here we report macro recall as "sensitivity_macro"
    metrics = {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "sensitivity_macro": float(r_macro),
    }

    labels = None if num_classes is None else list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return MetricsResult(metrics=metrics, confusion=cm)
