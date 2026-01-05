from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

import os
import numpy as np

NormalizeMode = Optional[Literal["true", "pred", "all"]]

def normalize_confusion(cm: np.ndarray, mode: NormalizeMode = "true") -> np.ndarray:
    """
    mode:
      - "true": normalize rows (each true class sums to 1)  [most common]
      - "pred": normalize cols (each predicted class sums to 1)
      - "all":  normalize by total sum
      - None:   return raw cm as float
    """
    cm = np.asarray(cm, dtype=np.float64)

    if mode is None:
        return cm

    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    if mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    if mode == "all":
        s = cm.sum()
        return cm / s if s != 0 else cm

    raise ValueError(f"Unknown normalize mode: {mode}")

def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_dir: str,
    model_name: str,
    normalize: NormalizeMode = "true",
    save_png: bool = True,
) -> dict:
    """
    Writes:
      - {model_name}_cm.npy
      - {model_name}_cm.csv
      - {model_name}_cm_norm_<mode>.npy
      - {model_name}_cm_norm_<mode>.csv
      - optional {model_name}_cm_norm_<mode>.png
      - {model_name}_labels.txt

    Returns a dict with file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square [C,C], got {cm.shape}")
    if len(labels) != cm.shape[0]:
        raise ValueError("labels length must match cm size")

    paths = {}

    labels_path = os.path.join(out_dir, f"{model_name}_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(str(lab) + "\n")
    paths["labels_txt"] = labels_path

    raw_npy = os.path.join(out_dir, f"{model_name}_cm.npy")
    np.save(raw_npy, cm)
    paths["cm_npy"] = raw_npy

    raw_csv = os.path.join(out_dir, f"{model_name}_cm.csv")
    _save_cm_csv(raw_csv, cm, labels)
    paths["cm_csv"] = raw_csv

    cm_norm = normalize_confusion(cm, normalize)
    mode = "none" if normalize is None else normalize

    norm_npy = os.path.join(out_dir, f"{model_name}_cm_norm_{mode}.npy")
    np.save(norm_npy, cm_norm)
    paths["cm_norm_npy"] = norm_npy

    norm_csv = os.path.join(out_dir, f"{model_name}_cm_norm_{mode}.csv")
    _save_cm_csv(norm_csv, cm_norm, labels)
    paths["cm_norm_csv"] = norm_csv

    if save_png:
        png_path = os.path.join(out_dir, f"{model_name}_cm_norm_{mode}.png")
        plot_confusion_matrix(cm_norm, labels, title=f"{model_name} (norm={mode})", out_path=png_path)
        paths["cm_norm_png"] = png_path

    return paths

def _save_cm_csv(path: str, cm: np.ndarray, labels: List[str]) -> None:
   
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(labels))
        for i, lab in enumerate(labels):
            writer.writerow([lab] + [float(x) for x in cm[i].tolist()])

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)  

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # write values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)
