"""
================================================================
 DecodeLabs - Artificial Intelligence Internship
 Project 4: Building the Machine's Optic Nerve
 PATH 2: Object Detection with MobileNet-SSD

 Engine: OpenCV's cv2.dnn module running a pre-trained
         MobileNet-SSD (Single Shot Detector) Caffe model,
         trained on the PASCAL VOC 20-class dataset.

 Pipeline:
   1. Blob Construction   - cv2.dnn.blobFromImage (mean subtraction +
                             resize to the network's 300x300 input)
   2. Forward Pass        - run the frozen network to get raw detections
   3. Coordinate Scaling  - convert normalized (0-1) box coordinates
                             back to real pixel coordinates
   4. Confidence Gate     - keep only detections >= 80% confidence
                             (Project 4's mandatory minimum standard)
   5. Visual Confirmation - draw labeled bounding boxes on the image
================================================================
"""

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.80   # Project 4 mandatory minimum (the "80% Gate")

# PASCAL VOC classes that MobileNet-SSD (chuanqi305 build) was trained on
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# A distinct color per class for the bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"


def load_model():
    """STEP 0: Load the frozen, pre-trained network (Transfer Learning)."""
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    return net


def detect_objects(net, image):
    """
    STEP 1-4: Build the blob, run inference, scale coordinates,
    and filter by the confidence gate.
    """
    (h, w) = image.shape[:2]

    # STEP 1: Blob Construction
    # - Resize to the network's required 300x300 input dimensions
    # - Mean subtraction (127.5) and scaling (0.007843 = 1/127.5)
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=0.007843, size=(300, 300),
        mean=127.5,
    )

    # STEP 2: Forward pass through the network
    net.setInput(blob)
    detections = net.forward()

    results = []
    # detections shape: [1, 1, num_detections, 7]
    # each row: [_, class_id, confidence, x1, y1, x2, y2] (all normalized 0-1)
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        # STEP 4: Confidence Gate - the "80% Threshold" from the slides
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(detections[0, 0, i, 1])
        label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

        # STEP 3: Coordinate Scaling - normalized -> actual pixel coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        results.append({
            "label": label,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
        })

    return results


def draw_detections(image, results):
    """STEP 5: Visual Confirmation - draw labeled, colored bounding boxes."""
    output = image.copy()
    for det in results:
        (x1, y1, x2, y2) = det["box"]
        class_id = CLASSES.index(det["label"]) if det["label"] in CLASSES else 0
        color = [int(c) for c in COLORS[class_id]]

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{det['label']}: {det['confidence']*100:.1f}%"
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            output, text, (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return output


def process_image(image_path, save_path):
    print("=" * 65)
    print(f"PATH 2: OBJECT DETECTION PIPELINE  ->  {image_path}")
    print("=" * 65)

    net = load_model()
    print("[1/4] Pre-trained MobileNet-SSD model loaded (Transfer Learning).")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = detect_objects(net, image)
    print(f"[2/4] Blob constructed + forward pass complete.")
    print(f"[3/4] Coordinates scaled to original image size.")
    print(f"[4/4] Confidence gate applied (>= {CONFIDENCE_THRESHOLD*100:.0f}%).")

    print("\n--- OUTPUT VALIDATION ---")
    if results:
        for det in sorted(results, key=lambda d: -d["confidence"]):
            print(
                f"  Detected: {det['label']:<12} "
                f"confidence: {det['confidence']*100:5.1f}%  "
                f"box: {det['box']}"
            )
    else:
        print("  No objects cleared the 80% confidence threshold.")

    annotated = draw_detections(image, results)
    cv2.imwrite(save_path, annotated)
    print(f"\nAnnotated image saved to: {save_path}")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    process_image("sample_images/sample_person.jpg", "output/detection_person.jpg")
    print()
    process_image("sample_images/sample_cat.jpg", "output/detection_cat.jpg")
