import random
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.emocatnets_v2 import emocatnetsv2_fer
from fer.xai.grad_cam import GradCAM


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


IMAGE_SIZE = 64
MAX_IMAGES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = Path("../src/fer/dataset/standardized/only_mtcnn_cropped/grey/png/test").resolve()
WEIGHTS_PATH = Path("../weights/emocatnetsv2/model_state_dict_emocat_v2.pt").resolve()
OUTPUT_DIR = Path("../xai_results/gradcam").resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Saving Grad-CAMs to:", OUTPUT_DIR)
print("Using device:", DEVICE)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

dataset = ImageFolder(DATASET_ROOT, transform=transform)
num_classes = len(dataset.classes)
per_class = MAX_IMAGES // num_classes

print(f"Classes ({num_classes}):", dataset.classes)
print(f"Images per class:", per_class)


class_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset):
    class_indices[label].append(idx)

selected_indices = []
for label, indices in class_indices.items():
    k = min(per_class, len(indices))
    selected_indices.extend(random.sample(indices, k))

random.shuffle(selected_indices)

print(f"Total selected images: {len(selected_indices)}")


model = emocatnetsv2_fer(size="tiny", in_channels=3,num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


cam = GradCAM(
    model=model,
    target_layers=[model.down3[1]]
)

print("Running Grad-CAM evaluation")


for i, idx in enumerate(tqdm(selected_indices)):
    model.zero_grad()
    x, y = dataset[idx]

    x = x.unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    heatmap = cam(x)[0]
    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

    img = x[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = np.clip(0.6 * img + 0.4 * heatmap_color, 0, 1)

    cls_name = dataset.classes[y].replace(" ", "_")

    class_dir = OUTPUT_DIR / cls_name
    class_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        class_dir / f"gradcam_{i:04d}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

print("Grad-CAM evaluation finished.")