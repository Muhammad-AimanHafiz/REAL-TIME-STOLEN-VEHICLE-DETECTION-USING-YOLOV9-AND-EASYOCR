import os
import cv2
import numpy as np
import easyocr
import re
from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64

app = Flask(__name__)

model = YOLO(r"C:/Users/User/PycharmProjects/PythonProject/runs/detect/yolov9s_number_plate2/weights/best.pt")

reader = easyocr.Reader(['en'], gpu=True)

output_folder = r"C:/Users/User/PycharmProjects/PythonProject/dataset/detection_result"
os.makedirs(output_folder, exist_ok=True)


def expand_box(x1, y1, x2, y2, pad_ratio, H, W):

    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return max(0, x1 - px), max(0, y1 - py), min(W - 1, x2 + px), min(H - 1, y2 + py)


def preprocess_plate(crop):

    up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return up, th, th_inv


def clean_plate(text: str) -> str:

    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    m = re.match(r'^([A-Z]{1,3})(\d{1,4})$', text)
    if m:
        return m.group(1) + m.group(2)

    candidates = re.findall(r'[A-Z0-9]+', text)
    return max(candidates, key=len) if candidates else ""


def ocr_plate(reader, crop):

    res = reader.readtext(
        crop,
        detail=1,
        paragraph=True,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/',
        text_threshold=0.9,
        low_text=0.5,
        mag_ratio=1.0)

    if not res:
        return ""

    def center(box):
        pts = np.array(box)
        return pts[:, 1].mean(), pts[:, 0].mean()

    res.sort(key=lambda r: (center(r[0])[0], center(r[0])[1]))
    raw = "".join([r[1] for r in res])
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())

    if not raw:
        tokens = [re.sub(r'[^A-Z0-9]', '', r[1].upper()) for r in res]
        tokens = [t for t in tokens if t]
        raw = max(tokens, key=len) if tokens else ""

    return clean_plate(raw)


@app.route("/detect", methods=["POST"])
def detect_plate():
    try:
        img_data = base64.b64decode(request.json["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        H, W, _ = img.shape
        results = model(img, verbose=False)

        all_detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    up, th, th_inv = preprocess_plate(crop)
                    text = (ocr_plate(reader, th) or
                            ocr_plate(reader, th_inv) or
                            ocr_plate(reader, up))

                    if text:
                        all_detections.append({
                            "plate": text,
                            "confidence": conf,
                            "xmin": float(x1),
                            "ymin": float(y1),
                            "xmax": float(x2),
                            "ymax": float(y2)
                        })

        return jsonify({
            "detections": all_detections,
            "image_width": W,
            "image_height": H
        })
    except Exception as e:
        return jsonify({"detections": [], "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)