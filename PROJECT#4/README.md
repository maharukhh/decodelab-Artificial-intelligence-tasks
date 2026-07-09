# Project 4: Building the Machine's Optic Nerve

**DecodeLabs Artificial Intelligence Internship — Industrial Training Kit (Batch 2026)**

## Overview

Project 4 moves from structured, tabular data (Projects 1–3) into
**unstructured visual data** — images. The mission: engineer a Python
pipeline that can "see" — either by reading text out of an image (OCR)
or by locating and naming physical objects in an image (object
detection) — with a **validated minimum confidence of 80%** on every
output.

Both paths described in the training slides are implemented here in
full, so this submission covers the complete Perception Matrix.

## Goal

> Engineer a Python script capable of ingesting raw visual data and
> extracting accurate, machine-readable intelligence.

| | Path 1: OCR | Path 2: Object Detection |
|---|---|---|
| **Objective** | Extract machine-readable text strings | Identify & locate physical entities |
| **Library** | `pytesseract` (Tesseract OCR engine) | `cv2.dnn` + MobileNet-SSD |
| **Pre-processing** | Grayscale → Blur → Deskew → Adaptive Threshold | 4D Blob construction (`blobFromImage`) |
| **Output** | Formatted text strings | (X, Y, W, H) bounding boxes + label |

## Files

```
path1_ocr_pipeline.py            -> Path 1: OCR pipeline
path2_object_detection.py        -> Path 2: Object detection pipeline 
models/
  MobileNetSSD_deploy.prototxt   -> Network architecture definition
  MobileNetSSD_deploy.caffemodel -> Pre-trained weights (Transfer Learning)
sample_images/
  sample_ocr_text.jpg            -> Synthetic noisy, tilted text image (OCR demo input)
  sample_person.jpg              -> Sample photo containing a person (detection demo input)
  sample_cat.jpg                 -> Sample photo containing a cat (detection demo input)
output/                           -> Generated after running the scripts (intermediate + final images)
README.md                        -> This file
```

## Path 1: OCR Pipeline — How It Works

Implements the exact "Logic Skeleton" from the slides:

1. **Grayscale Conversion** — collapses the 3-channel RGB matrix into a single intensity channel, removing distracting color data.
2. **Gaussian Blur** — smooths the image to eliminate micro-imperfections and noise before thresholding.
3. **Deskewing** — calculates the text block's rotation angle (via `cv2.minAreaRect`) and rotates the image back to a perfect horizontal baseline.
4. **Adaptive Thresholding (Otsu's method)** — forces every pixel to pure black or pure white using an automatically calculated cutoff, giving clean contours for character recognition.
5. **OCR + Confidence Gate** — runs `pytesseract.image_to_data` (PSM 6: uniform block of text) and keeps only words scoring **≥ 80% confidence**; anything below is logged as rejected/noise.

### Run it

```bash
pip install opencv-python pytesseract numpy
# Tesseract engine must also be installed on the system, e.g.:
#   sudo apt-get install tesseract-ocr
python3 path1_ocr_pipeline.py
```

### Reference result (on the included sample image)

```
Recognized text (>= 80% confidence): "DECODELABS Al 2026"
Average confidence of accepted words: 93.7%
  'DECODELABS'  ->  91.0%
  'Al'          ->  94.0%
  '2026'        ->  96.0%
```

The sample image was deliberately generated with Gaussian noise and a
6° tilt to prove the pre-processing pipeline (deskew + threshold) is
doing real work, not just reading a clean image.

## Path 2: Object Detection Pipeline — How It Works

Uses **Transfer Learning**: instead of training a neural network from
scratch, this loads a MobileNet-SSD model pre-trained on the PASCAL
VOC dataset (20 object classes: person, cat, dog, car, bus, etc.) and
runs inference directly — the "download a degree instead of training
from scratch" approach from the slides.

1. **Blob Construction** — `cv2.dnn.blobFromImage` resizes the input to the network's required 300×300 input, applies mean subtraction, and scales pixel values.
2. **Forward Pass** — the frozen network outputs normalized detection coordinates + a confidence score per detection.
3. **Coordinate Scaling** — normalized (0–1) coordinates are multiplied by the image's actual width/height to get real pixel bounding boxes.
4. **Confidence Gate (the "80% Threshold")** — any detection below 0.80 confidence is dropped, exactly as described in the slides' `if confidence >= 0.80: draw_box_and_label() else: drop_detection()` logic.
5. **Visual Confirmation** — accepted detections are drawn as labeled, colored bounding boxes on the output image.

### Run it

```bash
pip install opencv-python numpy
python3 path2_object_detection.py
```

### Reference results (on the included sample images)

| Image | Detected | Confidence |
|---|---|---|
| `sample_person.jpg` | person | 99.6% |
| `sample_cat.jpg` | cat | 100.0% |

Both comfortably clear the mandatory 80% confidence gate.

## Milestone Validation (per the Gatekeeper Rule)

| # | Requirement | Status |
|---|---|---|
| 1 | Library Integration (`pytesseract` **and** `cv2.dnn`) | ✅ Both implemented |
| 2 | Pre-Processing Integrity (Grayscale + Adaptive Thresholding) | ✅ Full pipeline in `path1_ocr_pipeline.py` |
| 3 | Accuracy Benchmarking (≥ 80% confidence) | ✅ Enforced in both scripts (`CONFIDENCE_THRESHOLD`) |
| 4 | Visual Confirmation (legible OCR string / labeled bounding boxes) | ✅ Saved to `output/` |

## Key Concepts Demonstrated

- **The IPO model for vision**: an image is a 3D array (Height × Width × Color Channels), each pixel 0–255.
- **Transfer Learning**: reusing a pre-trained model's learned features instead of training from scratch.
- **Image pre-processing**: grayscale, blur, deskew, adaptive (Otsu) thresholding.
- **Confidence scores & Softmax**: AI never "knows" — it estimates probabilities, and every output needs a confidence gate.
- **The 80% threshold trade-off**: high thresholds reduce false positives but risk more false negatives.
- **Bounding box decoding**: translating a network's normalized output back into real image coordinates.

## Notes on Sample Images

- `sample_ocr_text.jpg` is synthetically generated (clean text + Gaussian noise + 6° rotation) so the pre-processing steps have something real to correct.
- `sample_person.jpg` and `sample_cat.jpg` are the standard `astronaut` and `chelsea` sample images bundled with the `scikit-image` library (public test images, not scraped from the web) — swap in any of your own photos to test the pipeline further.

## Possible Extensions

- Try different `--psm` (Page Segmentation Mode) values in Tesseract for different layouts (single line, sparse text, etc.).
- Swap MobileNet-SSD for a newer model (YOLO, EfficientDet) for higher accuracy at the cost of more compute.
- Combine both paths: run object detection first, then OCR on any detected sign/label regions.

---
👩‍💻 Author Mahrukh

Robotics & Intelligent Systems Student

This project was completed as part of my Virtual Internship at DecodeLabs, where I gained hands-on experience with supervised machine learning and data classification using Python and scikit-learn.
