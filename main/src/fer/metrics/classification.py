from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

@dataclass
class MetricsResult:
    metrics: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    confusion: np.ndarray
 
    labels: List[str]

def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> MetricsResult:
    """
    Computes:
      - overall: accuracy, macro/weighted precision/recall/f1, micro f1, balanced accuracy (macro recall)
      - per-class: precision/recall/f1/support + sensitivity (recall) + specificity
      - confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        # infer from data
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        if len(class_names) != num_classes:
            raise ValueError(f"class_names length ({len(class_names)}) must equal num_classes ({num_classes}).")

    labels_idx = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    acc = float(accuracy_score(y_true, y_pred))

    p_c, r_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    balanced_acc = float(r_macro)


    total = float(np.sum(cm))

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total - tp - fn - fp)

        sensitivity = _safe_div(tp, tp + fn) 
        specificity = _safe_div(tn, tn + fp)

        per_class[name] = {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f1_c[i]),
            "support": float(support_c[i]),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,

        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),

        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),

        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),
    }

    return MetricsResult(
        metrics=metrics,
        per_class=per_class,
        confusion=cm,
        labels=class_names,
    )