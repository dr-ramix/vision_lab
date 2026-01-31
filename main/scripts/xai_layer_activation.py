import torch
import numpy as np
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.layer_activation import LayerActivationXAI



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DATASET_ROOT = "../src/fer/dataset/standardized/images_mtcnn_cropped_norm/test"
WEIGHTS_PATH = "../weights/resnet18fer/model_state_dict.pt"
OUTPUT_DIR = Path("../xai_results/layer_activation").resolve()

IMAGE_SIZE = 64
MAX_IMAGES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Saving results to:", OUTPUT_DIR)
print("Device:", DEVICE)



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = ImageFolder(DATASET_ROOT, transform=transform)
num_classes = len(dataset.classes)

print("Classes:", dataset.classes)



per_class = MAX_IMAGES // num_classes

class_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset):
    class_indices[label].append(idx)

selected_indices = []
for label, indices in class_indices.items():
    k = min(per_class, len(indices))
    selected_indices.extend(random.sample(indices, k))

random.shuffle(selected_indices)

print(f"Selected {len(selected_indices)} images "
      f"({per_class} per class)")



model = ResNet18FER(num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()



xai = LayerActivationXAI(
    model=model,
    target_layer=model.layer4[-1].conv2,  
    aggregation="mean"
)

print("Running Layer Activation XAI...")



for i, idx in enumerate(tqdm(selected_indices)):
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        activation_map = xai(x)[0]  # (H, W)

    activation_map = torch.relu(activation_map)
    activation_map = activation_map.cpu().numpy()

    activation_map = cv2.resize(
        activation_map, (IMAGE_SIZE, IMAGE_SIZE)
    )

    activation_map -= activation_map.min()
    activation_map /= activation_map.max() + 1e-8

    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * activation_map),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(
        heatmap_color, cv2.COLOR_BGR2RGB
    ) / 255.0

    overlay = np.clip(0.6 * img + 0.4 * heatmap_color, 0, 1)

    cls_name = dataset.classes[y].replace(" ", "_")
    class_dir = OUTPUT_DIR / cls_name
    class_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")

    axes[1].imshow(activation_map, cmap="jet")
    axes[1].set_title("Layer Activation")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        class_dir / f"layer_activation_{i:04d}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

print("Layer Activation XAI finished.")