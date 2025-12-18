
from __future__ import annotations
from typing import Callable
from torchvision import transforms

def build_train_transforms(cfg: dict) -> Callable:
    img_size = cfg.get("data", {}).get("img_size", 224)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

def build_eval_transforms(cfg: dict) -> Callable:
    img_size = cfg.get("data", {}).get("img_size", 224)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])