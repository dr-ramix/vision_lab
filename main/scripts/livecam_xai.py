import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from fer.pre_processing.basic_img_norms import BasicImageProcessor
from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper
from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.grad_cam import GradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "weights" / "resnet18fer" / "model_state_dict.pt"

IMG_SIZE = 64
DETECT_EVERY = 5
DETECT_MAX_SIZE = 480
SMOOTH_ALPHA = 0.65

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

# =========================================================
# MODEL
# =========================================================

model = ResNet18FER(num_classes=6, in_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

target_layers = [model.layer4[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

processor = BasicImageProcessor(target_size=(IMG_SIZE, IMG_SIZE))

# =========================================================
# FACE DETECTOR
# =========================================================

cropper = MTCNNFaceCropper(
    keep_all=True,
    min_prob=0.0,
    crop_scale=1.15,
    device=DEVICE,
)

# =========================================================
# HELPERS
# =========================================================

def resize_for_detection(frame_rgb):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIZE / max(h, w))
    if scale < 1.0:
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    return frame_rgb, scale


def preprocess_face(face_bgr):
    face_resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    proc = processor.process_bgr(face_resized)
    img = proc.normalized_rgb_vis.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)


def should_exit(name):
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        return True
    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False

# =========================================================
# STATE
# =========================================================

stable_boxes = None
cached_scale = 1.0

# =========================================================
# WEBCAM LOOP
# =========================================================

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

frame_idx = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---------------- FACE DETECTION ----------------
    if frame_idx % DETECT_EVERY == 0:
        small_rgb, scale = resize_for_detection(frame_rgb)
        pil_small = Image.fromarray(small_rgb)

        boxes, _ = cropper.mtcnn.detect(pil_small, landmarks=False)

        if boxes is not None:
            boxes = boxes.astype(np.float32)

            if stable_boxes is None or len(stable_boxes) != len(boxes):
                stable_boxes = boxes.copy()
            else:
                stable_boxes = (
                    SMOOTH_ALPHA * stable_boxes
                    + (1.0 - SMOOTH_ALPHA) * boxes
                )

            cached_scale = scale

    # =====================================================
    # FACE SELECTION + EMOTION
    # =====================================================

    emotion = "N/A"
    confidence = 0.0
    emotion_id = None

    face_tensor = None
    face_crop = None
    x1 = y1 = x2 = y2 = None

    if stable_boxes is not None:
        inv_scale = 1.0 / cached_scale

        # größtes Gesicht
        areas = []
        for box in stable_boxes:
            bx1, by1, bx2, by2 = (box * inv_scale).astype(int)
            areas.append(max(0, bx2 - bx1) * max(0, by2 - by1))

        idx = int(np.argmax(areas))
        box = stable_boxes[idx]

        x1, y1, x2, y2 = (box * inv_scale).astype(int)
        h, w = frame_bgr.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            face_crop = frame_bgr[y1:y2, x1:x2]
            face_tensor = preprocess_face(face_crop)

            with torch.no_grad():
                logits = model(face_tensor)
                probs = torch.softmax(logits, dim=1)

            emotion_id = probs.argmax().item()
            emotion = EMOTIONS[emotion_id]
            confidence = probs[0, emotion_id].item()

    
    blue_overlay = np.zeros_like(frame_bgr)
    blue_overlay[:] = (255, 0, 0)  # BGR Blue

    frame_bgr = cv2.addWeighted(frame_bgr, 0.25, blue_overlay, 0.75, 0)

    # =====================================================
    # FACE GRAD-CAM
    # =====================================================

    if face_tensor is not None and emotion_id is not None:
        grayscale_cam = cam(
            input_tensor=face_tensor,
            targets=[ClassifierOutputTarget(emotion_id)]
        )[0]  # (64,64)

        cam_face = cv2.resize(
            grayscale_cam,
            (x2 - x1, y2 - y1)
        )

        frame_bgr[y1:y2, x1:x2] = face_crop
        
        overlay_face = show_cam_on_image(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            cam_face,
            use_rgb=True
        )

        frame_bgr[y1:y2, x1:x2] = cv2.cvtColor(
            overlay_face, cv2.COLOR_RGB2BGR
        )

    # ---------------- DRAW BOXES ----------------
    if stable_boxes is not None:
        inv_scale = 1.0 / cached_scale
        for box in stable_boxes:
            bx1, by1, bx2, by2 = (box * inv_scale).astype(int)
            cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    # ---------------- LABEL ----------------
    cv2.putText(
        frame_bgr,
        f"{emotion} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("FER Webcam Demo (Face Grad-CAM)", frame_bgr)

    if should_exit("FER Webcam Demo (Face Grad-CAM)"):
        break

cap.release()
cv2.destroyAllWindows()