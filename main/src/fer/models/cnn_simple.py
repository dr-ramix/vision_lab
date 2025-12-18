import torch.nn as nn
from fer.models.factory import register_model

@register_model("cnn_simple")
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 7, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)
