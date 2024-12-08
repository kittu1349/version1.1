""" this version suggests the user all the pii types upon selection of choices blurs the options """

import easyocr
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os
import re
import warnings
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import argparse

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")

# Initialize the EasyOCR reader with GPU acceleration
reader = easyocr.Reader(['en'], gpu=True)

PII_PATTERNS = {
    "Permanent Account Number": r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    "Aadhaar Number": r'\b\d{12}\b|\b\d{4}\s\d{4}\s\d{4}\b',
    "Driving License Number": r'\b[A-Za-z0-9]{4}[- ]?\d{9,13}\b',
    "Passport Number": r'\b[A-Z]{1}[0-9]{7,9}\b',
    "Voter ID": r'[A-Z]{2,3}[0-9]{7,8}',
    "Credit Card": r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'
}

# Extract text from an image using EasyOCR
def extract_text_from_image(image):
    data = reader.readtext(np.array(image))
    text = ' '.join([text for _, text, _ in data])
    return text

# Detect PII in text
def detect_pii(text):
    detected_pii = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_pii[pii_type] = matches
    return detected_pii

# Blur bounding boxes around detected PII
def blur_pii_regions(image, pii_positions, blur_kernel_size=(75, 75), num_blur_passes=5):
    try:
        image = np.array(image)
        for (x1, y1, x2, y2) in pii_positions:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                kernel_size = (max(1, blur_kernel_size[0] | 1), max(1, blur_kernel_size[1] | 1))
                roi = image[y1:y2, x1:x2]
                for _ in range(num_blur_passes):
                    roi = cv2.GaussianBlur(roi, kernel_size, 0)
                dilated_roi = cv2.dilate(roi, None, iterations=1)
                image[y1:y2, x1:x2] = dilated_roi
        return Image.fromarray(np.uint8(image))
    except Exception as e:
        print(f"Error blurring PII regions: {e}")
        return image

# Calculate positions of detected PII in the image
def calculate_pii_positions(image, pii):
    positions = []
    try:
        data = reader.readtext(np.array(image))
        word_boxes = {}
        
        for bbox, word, _ in data:
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            word = word.strip()
            if word:
                word_boxes[word] = word_boxes.get(word, []) + [(x1, y1, x2, y2)]
        
        for pii_type, values in pii.items():
            for value in values:
                for word, boxes in word_boxes.items():
                    if fuzz.partial_ratio(value, word) > 80:
                        positions.extend(boxes)
                        
    except Exception as e:
        print(f"Error calculating positions: {e}")
    return positions

# Function to process and blur PII in a single image
def process_image(image, selected_pii_types=None):
    # Ensure the image is in RGB mode
    image = image.convert('RGB')
    
    # Extract text and detect PII
    text = extract_text_from_image(image)
    pii = detect_pii(text)
    
    if pii:
         if selected_pii_types is None:
            selected_pii_types = list(pii.keys())
        
        # Filter positions based on selected PII types
        selected_pii_positions = {ptype: pii[ptype] for ptype in selected_pii_types if ptype in pii}
        
        positions = calculate_pii_positions(image, selected_pii_positions)
        if positions:
            processed_image = blur_pii_regions(image, positions, blur_kernel_size=(75, 75), num_blur_passes=5)
            return processed_image
        else:
            print("No positions found for blurring.")
    return image

# Process PDF files
def process_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:\Program Files\poppler-24.07.0\Library\bin")
        print(f"Processing PDF: {pdf_path}")
        
        images_combined = []
        all_pii = {}

        # Collect PII from all pages
        for i, image in enumerate(images):
            print(f"Scanning page {i + 1} for PII...")

            text = extract_text_from_image(image)
            pii = detect_pii(text)
            
            # Merge detected PII from this page with the overall collection
            for key, value in pii.items():
                if key in all_pii:
                    all_pii[key].extend(value)
                else:
                    all_pii[key] = value
        
        if all_pii:
            # Suggest PII types based on the overall detection
            suggested_pii_types = list(all_pii.keys())
            print(f"Suggested PII types for blurring across the document: {', '.join(suggested_pii_types)}")
            
            # Wait for user input
            selected_pii_types = input("Enter the PII types to blur (comma-separated, or 'all' to blur all): ").strip()
            if selected_pii_types == 'all':
                selected_pii_types = suggested_pii_types
            else:
                selected_pii_types = [ptype.strip() for ptype in selected_pii_types.split(',')]
            
            # Process and blur PII in each page based on the selected types
            for i, image in enumerate(images):
                print(f"Processing page {i + 1}...")
                processed_image = process_image(image, selected_pii_types)
                images_combined.append(processed_image)
        
        if images_combined:
            processed_dir = 'processed'
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
                
            output_path = os.path.join(processed_dir, os.path.basename(pdf_path).replace(".", "_processed."))
            images_combined[0].save(output_path, save_all=True, append_images=images_combined[1:])
            print(f"Saved processed PDF as: {output_path}")

    except Exception as e:
        print(f"Error processing PDF: {e}")

def main():
    parser = argparse.ArgumentParser(description='Detect and blur PII in images and PDFs.')
    parser.add_argument('input', type=str, help='Path to an image file or PDF.')
    args = parser.parse_args()
    if os.path.isfile(args.input):
        _, ext = os.path.splitext(args.input)
        ext = ext.lower()
        if ext in ['.pdf']:
            process_pdf(args.input)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image = Image.open(args.input)
            processed_image = process_image(image)  # No need to set selected_pii_types explicitly
            if processed_image:
                processed_dir = 'processed'
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir)

                output_path = os.path.join(processed_dir, os.path.basename(args.input).replace(".", "_processed."))
                processed_image.save(output_path)
                print(f"Saved processed image as: {output_path}")
        else:
            print("Unsupported file format. Please provide a PDF or image file.")
    else:
        print(f"The file '{args.input}' does not exist.")

if __name__ == '__main__':
    main()
