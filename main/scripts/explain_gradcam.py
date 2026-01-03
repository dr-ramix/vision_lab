import torch
from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.grad_cam import GradCAM
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
import cv2
import numpy as np 

#from fer.dataset.dataloaders.dataloader import build_dataloaders, CLASS_ORDER, CLASS_TO_IDX

"""
class Config:
    images_root = Path("../src/fer/dataset/standardized/images_mtcnn_cropped_norm")
    batch_size = 64
    num_workers = 4
    num_classes = 6
    image_size = 64
"""

IMAGES_ROOT = Path("../src/fer/dataset/standardized/images_mtcnn_cropped_norm")
SPLIT = "test"
TARGET_CLASS = "anger"   
IMAGE_SIZE = 64

"""
cfg = Config()
"""

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(
    root=IMAGES_ROOT / SPLIT,
    transform=transform
)

class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

target_idx = class_to_idx[TARGET_CLASS]

for img, lbl in dataset:
    if lbl == target_idx:
        x = img.unsqueeze(0)
        label = lbl
        break

x.requires_grad_(True)

model = ResNet18FER(num_classes=6)
model.load_state_dict(torch.load("../weights/resnet18fer/model_state_dict.pt", map_location="cpu"))
model.eval()

"""
dls = build_dataloaders("../src/fer/dataset/standardized/images_mtcnn_cropped_norm/test/anger", batch_size=cfg.batch_size, num_workers=cfg.num_workers)
train_loader, val_loader, test_loader = dls.train, dls.val, dls.test
images, labels = next(iter(test_loader))
x = images[0].unsqueeze(0)
label = labels[0].item()
"""

target_class = "anger"
target_idx = dataset.class_to_idx[target_class]

for img, lbl in dataset:
    if lbl == target_idx:
        x = img.unsqueeze(0)   
        label = lbl
        break

x.requires_grad_(True)

cam = GradCAM(
    model=model,
    target_layers=[model.layer4[-1].conv2]
)

heatmap = cam(input_tensor=x)[0]
heatmap_resized = cv2.resize(heatmap, (64, 64))

img = x[0].permute(1, 2, 0).detach().numpy()
img = (img - img.min()) / (img.max() - img.min())

heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
heatmap_color = heatmap_color / 255.0

overlay = 0.6 * img + 0.4 * heatmap_color
overlay = np.clip(overlay, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))


axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(heatmap_resized, cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(f"../xai_results/grad_cam/gradcam.png", dpi=150, bbox_inches="tight")
plt.close()
