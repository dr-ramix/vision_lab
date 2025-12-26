
import torch.nn as nn

# -------------------------------
# Model
# -------------------------------
class CNNVanilla(nn.Module): #definieren uns unser neuronales Netzwerk
    def __init__(self, num_classes=6): #6 verschiedene Emotionen
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 # hier extrahieren wir uns Features aus den Bildern
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), #Aktivierungsfunktion (nicht-linear)
            nn.MaxPool2d(2),  # 64 ---> 32, reduziert die räumliche Größe

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (256,1,1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): #definiert, wie die Eingabe durch das Netzwerk fließt
        x = self.features(x)
        x = self.classifier(x)
        return x
