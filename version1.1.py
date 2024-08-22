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

# suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")

# if no gpu present then it may use default cpu
reader = easyocr.Reader(['en'], gpu=True)

PII_PATTERNS = {
    "Permanent Account Number": r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    "Aadhaar Number": r'\b\d{12}\b|\b\d{4}\s\d{4}\s\d{4}\b',
    "Driving License Number": r'\b[A-Za-z0-9]{4}[- ]?\d{9,13}\b',
    "Passport Number": r'\b[A-Z]{1}[0-9]{7,9}\b',
    "Voter ID": r'[A-Z]{2,3}[0-9]{7,8}',
    "Credit Card": r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'
}

# extracting text from an image using EasyOCR(GOOGLE)
def extract_text_from_image(image):
    data = reader.readtext(np.array(image))
    text = ' '.join([text for _, text, _ in data])
    return text

# detecting PII in all the parsed text
def detect_pii(text):
    detected_pii = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_pii[pii_type] = matches
    return detected_pii

# blurring detected bounding boxes around detected PII
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

# calculating positions for bounding boxes of detected PII in the image
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

# detect and blur PII in an image
def detect_and_blur_pii(image):
    try:
        # for PIL lib compatibility, check if the image is in RGB mode
        image = image.convert('RGB')
        
        # use of extract_text_from_image and detect_pii
        text = extract_text_from_image(image)
        pii = detect_pii(text)
        
        if pii:
            # puase and suggest detected pii to the user
            suggested_pii_types = list(pii.keys())
            print(f"Suggested PII types for blurring: {', '.join(suggested_pii_types)}")
            
            # wait for user input
            selected_pii_types = input("Enter the PII types to blur (comma-separated, or 'all' to blur all): ").strip().lower()
            if selected_pii_types == 'all':
                selected_pii_types = suggested_pii_types
            else:
                selected_pii_types = [ptype.strip() for ptype in selected_pii_types.split(',')]
            
            # finalize positions based on selected PII types
            selected_pii_positions = {ptype: pii[ptype] for ptype in selected_pii_types if ptype in pii}
            
            positions = calculate_pii_positions(image, selected_pii_positions)
            if positions:
                processed_image = blur_pii_regions(image, positions, blur_kernel_size=(75, 75), num_blur_passes=5)
                return processed_image
            else:
                print("No positions found for blurring.")
        else:
            print("No PII detected in text.")
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

# function that can process PDF files
def process_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:\Program Files\poppler-24.07.0\Library\bin")
        print(f"Processing PDF: {pdf_path}")
        
        images_combined = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i + 1}...")

            text = extract_text_from_image(image)
            pii = detect_pii(text)
            
            if pii:
                suggested_pii_types = list(pii.keys())
                print(f"Suggested PII types for blurring on page {i + 1}: {', '.join(suggested_pii_types)}")
                
                user_input = input(f"Do you want to blur PII on page {i + 1}? (yes/no): ").strip().lower()
                
                if user_input == 'yes':
                    processed_image = detect_and_blur_pii(image)
                    if processed_image:
                        images_combined.append(processed_image)
                    else:
                        images_combined.append(image)  # Add the original if processing fails
                else:
                    images_combined.append(image)  # Add the original if user chooses not to blur
            else:
                print(f"No PII detected on page {i + 1}.")
                images_combined.append(image)  # No PII detected, add the original image

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
            processed_image = detect_and_blur_pii(image)
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
