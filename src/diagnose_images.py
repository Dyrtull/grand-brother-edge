#!/usr/bin/env python3
"""
Image Diagnostic Tool
---------------------
Analyzes images to detect corruption, format issues, or detection failures.
Usage: python3 src/diagnose_images.py [optional_path_to_image]
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image

# Default path if none provided
DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "known_faces")

def analyze_image(filepath):
    """Comprehensive image analysis."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"✗ File does not exist: {filepath}")
        return False
    
    # 1. File Size Check
    file_size = os.path.getsize(filepath) / (1024 * 1024) # MB
    print(f"File size: {file_size:.2f} MB")
    if file_size > 5:
        print("  ⚠ WARNING: Large file size (may slow down loading)")
    
    # 2. PIL Load Test (Metadata & Color Mode)
    try:
        pil_img = Image.open(filepath)
        print(f"✓ PIL load: SUCCESS")
        print(f"  - Dimensions: {pil_img.size[0]}x{pil_img.size[1]}")
        print(f"  - Mode: {pil_img.mode}")
        
        if pil_img.mode != 'RGB':
            print(f"  ⚠ WARNING: Non-RGB mode ({pil_img.mode}). This often crashes dlib.")
            
        pil_array = np.array(pil_img)
    except Exception as e:
        print(f"✗ PIL load FAILED: {e}")
        return False
    
    # 3. OpenCV Load Test
    try:
        cv_img = cv2.imread(filepath)
        if cv_img is not None:
            print(f"✓ OpenCV load: SUCCESS")
        else:
            print(f"✗ OpenCV load FAILED: Returned None (Corrupt header?)")
            return False
    except Exception as e:
        print(f"✗ OpenCV load FAILED: {e}")
        return False
    
    # 4. Face Detection Test (The Real Test)
    try:
        import face_recognition
        fr_img = face_recognition.load_image_file(filepath)
        
        # Test Strategy A: Standard
        print("\n--- Detection Strategy Tests ---")
        print("1. Standard HOG (upsample=1)...", end=" ")
        faces_1 = face_recognition.face_locations(fr_img, model='hog', number_of_times_to_upsample=1)
        print(f"{len(faces_1)} faces")
        
        # Test Strategy B: Sensitive
        print("2. Sensitive HOG (upsample=2)...", end=" ")
        faces_2 = face_recognition.face_locations(fr_img, model='hog', number_of_times_to_upsample=2)
        print(f"{len(faces_2)} faces")
        
        # Result
        if faces_1 or faces_2:
            print(f"\n✓ PASS: Face detected!")
            return True
        else:
            print(f"\n✗ FAIL: No face detected.")
            print("  Recommendation: Use prepare_images.py or retake photo.")
            return False
            
    except Exception as e:
        print(f"✗ Face Recognition crash: {e}")
        return False

def main():
    # Determine target
    target = DEFAULT_DIR
    if len(sys.argv) > 1:
        target = sys.argv[1]

    print(f"Target: {target}")

    if os.path.isfile(target):
        analyze_image(target)
    elif os.path.isdir(target):
        files = [f for f in os.listdir(target) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print("No images found in directory.")
        for f in files:
            analyze_image(os.path.join(target, f))
    else:
        print("Invalid path.")

if __name__ == "__main__":
    main()
