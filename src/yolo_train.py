import torch
from ultralytics import YOLO
import easyocr
import cv2

def main():
    print(torch.cuda.is_available())       # should return True
    print(torch.cuda.get_device_name(0))   # shows GPU model

    # Load YOLOv9 model
    model = YOLO('../yolov9s.pt')

    # Start training
    model.train(
        data='C:/Users/User/PycharmProjects/PythonProject/dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        name='yolov9s_number_plate',
        workers=0,                          
        optimizer='SGD',
        cos_lr=True,
        patience=50,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        auto_augment='randaugment',
        val=True,
        device='cuda'
    )

    # Run validation and apply EasyOCR
    val_results = model.val()  # This runs validation (after training epochs)
    print("\n✅ Final Accuracy Report:")
    print(f"  - mAP50: {val_results.box.map50:.4f}")
    print(f"  - Precision: {val_results.box.precision:.4f}")
    print(f"  - Recall: {val_results.box.recall:.4f}")

    # Now, applying OCR to detected plates in the validation results
    apply_ocr_to_detections(val_results)

def apply_ocr_to_detections(val_results):
    # Initialize EasyOCR
    ocr = easyocr.Reader(['en'])

    # Iterate through validation results (detected plates)
    for result in val_results.xywh:  # Assuming this contains bounding boxes of detected plates
        # Assuming that val_results.xywh gives you the detected bounding boxes in format (x1, y1, x2, y2)
        xyxy = result[0]  # Get coordinates for each detection (x1, y1, x2, y2)

        # Use the coordinates to crop the image (number plate region)
        img_path = result[1]  # Get image path (this depends on how your val_results are structured)
        img = cv2.imread(img_path)

        # Extract the bounding box
        x1, y1, x2, y2 = map(int, xyxy)
        cropped_img = img[y1:y2, x1:x2]

        # Run OCR on the cropped number plate
        ocr_result = ocr.readtext(cropped_img)

        # Display OCR result for each detected number plate
        for result in ocr_result:
            plate_text = result[1]  # Extract text
            print(f"Detected Plate: {plate_text}")

# REQUIRED on Windows
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

# To run:
# python -m src.yolo_train
