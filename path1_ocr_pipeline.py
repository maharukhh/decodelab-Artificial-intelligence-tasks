"""
================================================================
 DecodeLabs - Artificial Intelligence Internship
 Project 4: Building the Machine's Optic Nerve
 PATH 1: Optical Character Recognition (OCR)

 Engine: pytesseract (Python wrapper for Google's Tesseract OCR)

 Pipeline (per the Logic Skeleton in the training slides):
   1. Grayscale Conversion   - collapse 3D RGB matrix to 1D intensity
   2. Gaussian Blur          - smooth out micro-noise/artifacts
   3. Deskewing              - snap tilted text back to horizontal
   4. Adaptive Thresholding  - force a clean black/white binary image
   5. OCR + Confidence Gate  - extract text, keep only >= 80% confidence
================================================================
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

CONFIDENCE_THRESHOLD = 80.0   # Project 4 mandatory minimum (Gatekeeper Rule)


# ----------------------------------------------------------------
# STEP 1: Grayscale Conversion
# ----------------------------------------------------------------
def to_grayscale(image):
    """Collapses the 3-channel (R,G,B) matrix into a single intensity channel."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------
# STEP 2: Gaussian Blur
# ----------------------------------------------------------------
def denoise(gray_image, kernel_size=(5, 5)):
    """Smooths the image to remove chromatic noise and small artifacts."""
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


# ----------------------------------------------------------------
# STEP 3: Deskewing
# ----------------------------------------------------------------
def deskew(gray_image):
    """
    Calculates the rotation angle of the text block and rotates the
    image back to a perfect horizontal baseline.
    """
    # Invert + threshold so text pixels are white on a black background,
    # which is what cv2.minAreaRect expects for finding text orientation.
    inverted = cv2.bitwise_not(gray_image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return gray_image, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect angle convention correction
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


# ----------------------------------------------------------------
# STEP 4: Adaptive Thresholding (Otsu's method)
# ----------------------------------------------------------------
def binarize(gray_image):
    """
    Forces every pixel to choose a side: pure black or pure white.
    IF pixel_intensity >= cutoff THEN 255 (white) ELSE 0 (black)
    Otsu automatically calculates the optimal cutoff value.
    """
    cutoff, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, cutoff


# ----------------------------------------------------------------
# STEP 5: OCR + Confidence Gate
# ----------------------------------------------------------------
def run_ocr(processed_image, psm=6):
    """
    Runs Tesseract OCR and returns only words that clear the
    Project 4 mandatory 80% confidence threshold.
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        processed_image, config=config, output_type=Output.DICT
    )

    accepted_words = []
    rejected_words = []

    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        word = word.strip()
        if not word:
            continue
        if conf >= CONFIDENCE_THRESHOLD:
            accepted_words.append((word, conf))
        else:
            rejected_words.append((word, conf))

    return accepted_words, rejected_words


# ----------------------------------------------------------------
# FULL PIPELINE
# ----------------------------------------------------------------
def process_image(image_path, save_prefix="output/ocr"):
    print("=" * 65)
    print(f"PATH 1: OCR PIPELINE  ->  {image_path}")
    print("=" * 65)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 1: Grayscale
    gray = to_grayscale(image)
    cv2.imwrite(f"{save_prefix}_1_grayscale.jpg", gray)
    print("[1/4] Grayscale conversion complete.")

    # Step 2: Denoise
    blurred = denoise(gray)
    cv2.imwrite(f"{save_prefix}_2_blurred.jpg", blurred)
    print("[2/4] Gaussian blur applied.")

    # Step 3: Deskew
    deskewed, angle = deskew(blurred)
    cv2.imwrite(f"{save_prefix}_3_deskewed.jpg", deskewed)
    print(f"[3/4] Deskewed by {angle:.2f} degrees.")

    # Step 4: Adaptive threshold (binarize)
    binary, cutoff = binarize(deskewed)
    cv2.imwrite(f"{save_prefix}_4_binary.jpg", binary)
    print(f"[4/4] Adaptive threshold applied (Otsu cutoff = {cutoff:.1f}).")

    # Step 5: OCR with confidence gate
    accepted, rejected = run_ocr(binary)

    print("\n--- OUTPUT VALIDATION ---")
    if accepted:
        full_text = " ".join(w for w, c in accepted)
        avg_conf = sum(c for w, c in accepted) / len(accepted)
        print(f"Recognized text (>= {CONFIDENCE_THRESHOLD:.0f}% confidence):")
        print(f'  "{full_text}"')
        print(f"Average confidence of accepted words: {avg_conf:.1f}%")
        print(f"Words accepted: {len(accepted)} | Words rejected (low-confidence): {len(rejected)}")
        for w, c in accepted:
            print(f"    '{w}'  ->  {c:.1f}%")
    else:
        print("No text cleared the 80% confidence threshold.")

    if rejected:
        print("\nRejected (below threshold, likely noise):")
        for w, c in rejected:
            print(f"    '{w}'  ->  {c:.1f}%")

    return accepted, rejected


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    process_image("sample_images/sample_ocr_text.jpg", save_prefix="output/ocr")
