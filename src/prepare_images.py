#!/usr/bin/env python3
"""
Image Preparation Utility
--------------------------
A standalone tool to sanitize reference photos for face recognition.
- Converts images to standard RGB (removes weird color profiles).
- Resizes extremely large images (optimizing loading speed).
- Saves as high-compatibility JPEG.

Usage:
    python3 src/prepare_images.py <input_folder> <output_folder>
"""

import sys
import os
from PIL import Image
import numpy as np

def prepare_image(input_path, output_path, max_size=1200, quality=95):
    """
    Sanitizes an image: converts to RGB, resizes if needed, and saves.
    """
    try:
        print(f"  Processing: {os.path.basename(input_path)}")
        
        # Load with PIL (handles metadata better than OpenCV)
        img = Image.open(input_path)
        orig_w, orig_h = img.size
        
        # Force conversion to standard RGB (fixes dlib crashes)
        if img.mode != 'RGB':
            print(f"    Converting mode {img.mode} -> RGB")
            img = img.convert('RGB')
        
        # Smart Resize: Downscale only if the image is huge
        if max(orig_w, orig_h) > max_size:
            scale = max_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            print(f"    Resizing: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            print(f"    Size OK ({orig_w}x{orig_h})")
        
        # Save sanitized file
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
        final_size = os.path.getsize(output_path) / 1024  # KB
        print(f"    ✓ Saved: {output_path} ({final_size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("IMAGE PREPARATION UTILITY")
    print("="*70)
    
    # Define paths (Edit these if your uploads are elsewhere)
    # Assuming images are currently in your downloads or a specific upload folder
    # For this run, we assume you put the raw files in a folder named 'raw_photos'
    # If not, change 'raw_photos' to the path where your 70028.jpg files are.
    input_dir = "raw_photos" 
    output_dir = "known_faces" 
    
    # Override with command line arguments if provided
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

    print(f"\nConfiguration:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"\n[ERROR] Input directory '{input_dir}' not found.")
        print("Please create it and put your 700xx.jpg photos inside.")
        return
    
    # Create output directory if missing
    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not image_files:
        print(f"\n[WARN] No images found in '{input_dir}'")
        return
    
    print(f"\nFound {len(image_files)} images to process.\n")
    
    success_count = 0
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        
        # Keep original filename for now, you will rename them manually after
        output_path = os.path.join(output_dir, filename)
        
        if prepare_image(input_path, output_path):
            success_count += 1
        print("-" * 30)
    
    print("="*70)
    print(f"COMPLETED: {success_count}/{len(image_files)} images processed.")
    print(f"Location: {os.path.abspath(output_dir)}")
    print("="*70)
    
    if success_count > 0:
        print("\nCRITICAL NEXT STEP:")
        print("Go to the 'known_faces' folder and rename these files to:")
        print("  - Andres_1.jpg")
        print("  - Andres_2.jpg")
        print("  - Andres_3.jpg")

if __name__ == "__main__":
    main()
