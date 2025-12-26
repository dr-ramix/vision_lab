import torch
import torch.nn as nn
import cv2


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


# -----------------------------------
# Setup
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNVanilla(num_classes=6).to(device)
model.load_state_dict(torch.load("emotion_cnn_mtcnn.pth", map_location=device))
model.eval()

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# -----------------------------------
# OpenCV Webcam
# -----------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------
    # PREPROCESSING (NUR KOMMENTARE!)
    # -----------------------------------

    # 1. Gesicht erkennen (z.B. mit MTCNN / Haar Cascade)
    # 2. Gesicht aus dem Frame croppen
    # 3. In RGB umwandeln (OpenCV liefert BGR)
    # 4. Auf 64x64 resize
    # 5. In float umwandeln und normalisieren
    # 6. Shape ändern: (H,W,C) -> (C,H,W)
    # 7. Batch-Dimension hinzufügen: (1,C,H,W)
    # 8. Tensor auf GPU/CPU verschieben

    # -----------------------------------
    # DUMMY INPUT (nur damit Code läuft)
    # ---> wird durch echtes Preprocessing ersetzt
    # -----------------------------------
    dummy_input = torch.zeros((1, 3, 64, 64)).to(device)

    # -----------------------------------
    # CNN Inference
    # -----------------------------------
    with torch.no_grad():
        outputs = model(dummy_input)
        pred = outputs.argmax(dim=1).item()
        emotion = EMOTIONS[pred]

    # -----------------------------------
    # Anzeige
    # -----------------------------------
    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------
# Cleanup
# -----------------------------------
cap.release()
cv2.destroyAllWindows()
