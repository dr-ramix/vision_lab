import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

import fer.models.cnn_simple # ensures registration (if you didn't auto-import in __init__)
from fer.models import build_model
from fer.engine import fit, TrainConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "data": {"img_size": 224},
        "model": {"name": "cnn_simple", "num_classes": 7, "in_ch": 3},
    }

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = FakeData(size=128, image_size=(3, 224, 224), num_classes=7, transform=tfm)
    val_ds = FakeData(size=64, image_size=(3, 224, 224), num_classes=7, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = build_model(cfg)

    tcfg = TrainConfig(
        epochs=2,
        lr=3e-4,
        out_dir="outputs/dummy_run",
        num_classes=7,
        save_best_on="f1_macro",
    )

    fit(model, train_loader, val_loader, device, tcfg)
    print("Done. Check outputs/dummy_run/{last.pth,best.pth}")

if __name__ == "__main__":
    main()
