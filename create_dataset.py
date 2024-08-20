from pytesseract import image_to_data, pytesseract
from PIL import Image
import numpy as np
import cv2
import os
import re
import argparse
import shutil
from fuzzywuzzy import fuzz

# Path to Tesseract executable (only needed if not in system PATH)
pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Detect PII in text with fuzzy matching
def detect_pii(text):
    pii_patterns = {
        "Permanent Account Number": r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
        "PAN Number": r'\b[A-Z]{3,5}[0-9]{3,5}[A-Z]{0,2}\b',
    }
    detected_pii = {}
    for pii_type, pattern in pii_patterns.items():
        for line in text.split('\n'):
            for match in re.finditer(pattern, line):
                if fuzz.partial_ratio(match.group(), line) > 80:
                    detected_pii.setdefault(pii_type, []).append(match.group())
    return detected_pii

# Blur bounding boxes around detected PII
def blur_pii_regions(image, pii_positions, blur_kernel_size=(35, 35), num_blur_passes=2):
    try:
        image = np.array(image)
        for (x1, y1, x2, y2) in pii_positions:
            kernel_size = (max(1, blur_kernel_size[0] | 1), max(1, blur_kernel_size[1] | 1))
            roi = image[y1:y2, x1:x2]
            for _ in range(num_blur_passes):
                roi = cv2.GaussianBlur(roi, kernel_size, 0)
            dilated_roi = cv2.dilate(roi, None, iterations=1)
            image[y1:y2, x1:x2] = dilated_roi
        return Image.fromarray(image)
    except Exception as e:
        print(f"Error blurring PII regions: {e}")
        return image

# Calculate positions of detected PII in the image
def calculate_pii_positions(image, pii):
    positions = []
    try:
        data = image_to_data(image, output_type='dict')
        num_boxes = len(data['text'])
        word_boxes = {}
        for i in range(num_boxes):
            if int(data['conf'][i]) > 0:
                word = data['text'][i].strip()
                if word:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    if word not in word_boxes:
                        word_boxes[word] = []
                    word_boxes[word].append((x1, y1, x2, y2))
        for pii_type, values in pii.items():
            for value in values:
                if value in word_boxes:
                    for box in word_boxes[value]:
                        x1, y1, x2, y2 = box
                        positions.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"Error calculating positions: {e}")
    return positions

# Detect and blur PII in an image
def process_image(image_path, output_dir):
    try:
        image = Image.open(image_path)
        data = image_to_data(image, output_type='dict')
        text = ' '.join(data['text'])
        pii = detect_pii(text)
        positions = []
        if pii:
            positions = calculate_pii_positions(image, pii)
        # Save original and blurred images
        original_output_path = os.path.join(output_dir, 'not_blurred', os.path.basename(image_path))
        blurred_output_path = os.path.join(output_dir, 'blurred', os.path.basename(image_path))
        if positions:
            blurred_image = blur_pii_regions(image, positions)
            blurred_image.save(blurred_output_path)
        else:
            shutil.copy(image_path, original_output_path)
    except Exception as e:
        print(f'Error processing {image_path}: {e}')

# Process directory of images
def process_directory(directory_path, output_dir):
    os.makedirs(os.path.join(output_dir, 'blurred'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'not_blurred'), exist_ok=True)
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, file)
            print(f'\nProcessing {file}...')
            process_image(image_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Create a dataset by detecting and blurring PII in images.')
    parser.add_argument('input', type=str, help='Path to a directory containing images.')
    parser.add_argument('output', type=str, help='Path to the output directory for the dataset.')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        print(f"The directory '{args.input}' does not exist.")

if __name__ == '__main__':
    main()
