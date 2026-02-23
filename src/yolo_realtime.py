import os
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO


# ------------------------
# Model + output folder
# ------------------------
model = YOLO(
    r"C:/Users/User/PycharmProjects/PythonProject/runs/detect/yolov9s_number_plate24/weights/best.pt"
)

output_folder = r"C:/Users/User/PycharmProjects/PythonProject/dataset/detection_result"
os.makedirs(output_folder, exist_ok=True)

# One EasyOCR reader (reuse for speed)
reader = easyocr.Reader(['en'], gpu=True)   # set gpu=False if GPU gives error


# ------------------------
# Helper functions
# ------------------------
def expand_box(x1, y1, x2, y2, pad_ratio, H, W):
    """Add padding around YOLO box but stay inside image."""
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return max(0, x1 - px), max(0, y1 - py), min(W - 1, x2 + px), min(H - 1, y2 + py)


def preprocess_plate(crop):
    """Make crop more OCR-friendly (resize, denoise, contrast, binarize)."""
    up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return up, th, th_inv


def clean_plate(text: str) -> str:
    """Post-process text to fit plate format, fix common mistakes."""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())   # keep A–Z0–9 only

    # Example MY plate patterns: letters(1–3) + digits(1–4)
    m = re.match(r'^([A-Z]{1,3})(\d{1,4})$', text)
    if m:
        return m.group(1) + m.group(2)   # already no space: ABC1234

    # Fallback: choose the longest alphanumeric chunk
    candidates = re.findall(r'[A-Z0-9]+', text)
    return max(candidates, key=len) if candidates else ""


def ocr_plate(reader, crop):
    """Run EasyOCR on a crop and return cleaned plate text."""
    res = reader.readtext(
        crop,
        detail=1,
        paragraph=True,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/',
        text_threshold=0.9, # 0.9 , 0.5 , 1.0
        low_text=0.5, # 0.3
        mag_ratio=1.0, # 2.0
    )

    if not res:
        return ""

    # Sort detections: top→bottom, left→right
    def center(box):
        pts = np.array(box)
        return pts[:, 1].mean(), pts[:, 0].mean()

    res.sort(key=lambda r: (center(r[0])[0], center(r[0])[1]))

    # Merge all strings
    raw = "".join([r[1] for r in res])
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())

    if not raw:
        # fallback – pick longest cleaned token
        tokens = [re.sub(r'[^A-Z0-9]', '', r[1].upper()) for r in res]
        tokens = [t for t in tokens if t]
        raw = max(tokens, key=len) if tokens else ""

    return clean_plate(raw)


# ------------------------
# Real-time loop (webcam)
# ------------------------
# 0 = default webcam. Replace with video path if you want to use a video file.
#cap = cv2.VideoCapture("C:/Users/User/PycharmProjects/PythonProject/dataset/Video_test/VID_20251118_141517_HDR10PLUS.mp4")

#External USB webcam
cap = cv2.VideoCapture(1)

#Laptop’s built-in camera
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    img = frame.copy()
    H, W = img.shape[:2]

    # Run YOLO on current frame
    results = model(img, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

# ------------------------
# Main loop over YOLO detections
# ------------------------
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, pad_ratio=0.12, H=H, W=W)
        crop = img[y1:y2, x1:x2]

        # Skip invalid crops
        if crop.size == 0:
            continue

        up, th, th_inv = preprocess_plate(crop)

        # Try several variants and keep first non-empty
        text = (
            ocr_plate(reader, th)
            or ocr_plate(reader, th_inv)
            or ocr_plate(reader, up)
        )

        if text:
            print(f"Detected Plate: {text}")
            # draw bbox + label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (54, 204, 54),
                3,
            )

    # Show live result
    cv2.imshow("Real-time Plate Detection", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # quit
        break
    elif key == ord('s'):
        # save current frame
        frame_id += 1
        save_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
        cv2.imwrite(save_path, img)
        print(f"💾 Saved frame to {save_path}")

cap.release()
cv2.destroyAllWindows()
#Press q to quit, s to save current frame into detection_result

#run: python -m src.yolo_realtime
