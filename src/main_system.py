#!/usr/bin/env python3

"""

Grand Brother - Edge AI Identity Recognition System (IMPROVED)

---------------------------------------------------------------

A hybrid Computer Vision system combining YOLOv8 (Human Detection) 

and dlib (Face Recognition) running on Raspberry Pi 5.



Author: Andres (PhD Candidate)

Date: February 2026

Improved Version with enhanced error handling and performance optimizations

"""



import cv2

import face_recognition

from ultralytics import YOLO

import numpy as np

import time

import os

import csv

from datetime import datetime

from typing import List, Tuple, Dict

from collections import deque



# --- CONFIGURATION ---

CONF_THRESHOLD = 0.5        # YOLO confidence threshold

LOG_COOLDOWN = 5            # Seconds between logs for the same person

KNOWN_FACES_DIR = "../known_faces"  # Directory containing reference images

LOG_DIR = "../logs"         # Directory to save access logs

CAMERA_WIDTH = 640          # Camera resolution width

CAMERA_HEIGHT = 480         # Camera resolution height

FPS_BUFFER_SIZE = 30        # Number of frames for FPS averaging

MIN_ROI_SIZE = 20           # Minimum ROI dimension to process

FACE_DISTANCE_THRESHOLD = 0.6  # Lower = stricter matching (0.6 is recommended)



# --- GLOBAL STATE ---

last_seen: Dict[str, float] = {}  # Store last detection timestamps: {name: time}

fps_buffer = deque(maxlen=FPS_BUFFER_SIZE)  # Rolling buffer for FPS calculation





def load_known_faces(directory: str) -> Tuple[List[np.ndarray], List[str]]:

    """

    Iterates through the 'known_faces' directory, encodes every image found,

    and returns lists of encodings and names.

    

    Args:

        directory: Path to directory containing reference face images

        

    Returns:

        Tuple of (encodings list, names list)

    """

    encodings = []

    names = []

    

    print(f"[INIT] Loading known faces from {directory}...")

    

    if not os.path.exists(directory):

        print(f"[ERROR] Directory '{directory}' not found. Creating it...")

        os.makedirs(directory, exist_ok=True)

        print(f"[INFO] Please add reference images to {directory}")

        return encodings, names



    image_files = [f for f in os.listdir(directory) 

                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    

    if not image_files:

        print(f"[WARNING] No image files found in {directory}")

        return encodings, names



    for filename in image_files:

        path = os.path.join(directory, filename)

        try:

            # Load image

            image = face_recognition.load_image_file(path)

            # Encode face (assume only 1 face per reference image)

            face_encodings_list = face_recognition.face_encodings(image)

            

            if not face_encodings_list:

                print(f"  [WARNING] No face found in {filename}. Skipping.")

                continue

                

            encoding = face_encodings_list[0]

            encodings.append(encoding)

            

            # Use filename without extension as name (e.g., 'Andres.jpg' -> 'Andres')

            name = os.path.splitext(filename)[0]

            names.append(name)

            print(f"  > Loaded: {name}")

            

        except Exception as e:

            print(f"  [ERROR] Could not process {filename}: {e}")

                

    print(f"[INIT] Total known faces loaded: {len(names)}")

    return encodings, names





def log_event(name: str, confidence: float) -> None:

    """

    Logs the detection event to a daily CSV file if the cooldown period has passed.

    

    Args:

        name: Identified person's name

        confidence: Recognition confidence score

    """

    now = time.time()

    

    # Check cooldown (Anti-spam filter)

    if name in last_seen and (now - last_seen[name] <= LOG_COOLDOWN):

        return

        

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date_str = datetime.now().strftime("%Y%m%d")

    

    print(f"[EVENT] {name} detected! Logging to CSV...")

    

    # Ensure log directory exists

    os.makedirs(LOG_DIR, exist_ok=True)

    filename = os.path.join(LOG_DIR, f"access_log_{date_str}.csv")

    file_exists = os.path.isfile(filename)

    

    try:

        with open(filename, mode='a', newline='') as file:

            writer = csv.writer(file)

            # Write header if new file

            if not file_exists:

                writer.writerow(["Timestamp", "ID", "Confidence", "Model"])

            

            writer.writerow([timestamp_str, name, f"{confidence:.2f}", "Hybrid-YOLO-dlib"])

            

        # Update last seen time

        last_seen[name] = now

        

    except Exception as e:

        print(f"[ERROR] Failed to write log: {e}")





def identify_face(face_encoding: np.ndarray, 

                 known_encodings: List[np.ndarray], 

                 known_names: List[str]) -> Tuple[str, float]:

    """

    Identifies a face by comparing its encoding against known faces.

    Uses face distance for more accurate matching.

    

    Args:

        face_encoding: The encoding of the face to identify

        known_encodings: List of known face encodings

        known_names: List of names corresponding to known encodings

        

    Returns:

        Tuple of (name, confidence) where confidence is 1 - distance

    """

    if not known_encodings:

        return "Unknown", 0.0

    

    # Calculate face distances (Euclidean distance in encoding space)

    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    

    # Find the best match

    best_match_index = np.argmin(face_distances)

    min_distance = face_distances[best_match_index]

    

    # Check if the best match is below our threshold

    if min_distance < FACE_DISTANCE_THRESHOLD:

        name = known_names[best_match_index]

        confidence = 1.0 - min_distance  # Convert distance to confidence

        return name, confidence

    

    return "Unknown", 0.0





def calculate_fps(curr_time: float, prev_time: float) -> float:

    """

    Calculates smoothed FPS using a rolling average.

    

    Args:

        curr_time: Current timestamp

        prev_time: Previous frame timestamp

        

    Returns:

        Smoothed FPS value

    """

    frame_time = curr_time - prev_time

    if frame_time > 0:

        fps_buffer.append(1.0 / frame_time)

    

    return sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0.0





def main():

    """Main execution function with improved error handling."""

    

    # 1. Load Models

    print("[INIT] Loading YOLOv8 Nano model...")

    try:

        yolo_model = YOLO('yolov8n.pt')

    except Exception as e:

        print(f"[ERROR] Failed to load YOLO model: {e}")

        print("[INFO] Make sure 'yolov8n.pt' is in the current directory or will be auto-downloaded")

        return

    

    # 2. Load Knowledge Base

    known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

    

    if not known_face_encodings:

        print("[WARNING] No known faces loaded. System will only detect persons without identification.")

    

    # 3. Setup Camera

    print("[INIT] Initializing camera...")

    cap = cv2.VideoCapture(0)

    

    if not cap.isOpened():

        print("[ERROR] Could not open camera. Please check camera connection.")

        return

    

    # Set camera properties

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    

    # Verify camera settings

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"[INIT] Camera resolution: {int(actual_width)}x{int(actual_height)}")



    print("[RUN] System started. Press 'q' to exit.")

    

    prev_time = time.time()

    frame_count = 0



    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("[WARNING] Failed to grab frame. Retrying...")

                time.sleep(0.1)

                continue



            frame_count += 1



            # --- STEP 1: OBJECT DETECTION (YOLO) ---

            # Detect 'person' class only (class index 0)

            # Using stream=False to get complete results

            results = yolo_model.predict(

                frame, 

                conf=CONF_THRESHOLD, 

                classes=[0],  # Person class

                verbose=False,

                device='cpu'  # Explicit CPU usage for Raspberry Pi

            )

            

            # Convert frame to RGB (required by face_recognition/dlib)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



            # Process each detected person

            for box in results[0].boxes:

                # Get coordinates

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                

                # Boundary checking to ensure ROI is within frame

                x1, y1 = max(0, x1), max(0, y1)

                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                

                # Extract Region of Interest (ROI) - The person

                # np.ascontiguousarray ensures proper memory layout for dlib

                person_roi = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

                

                # Skip artifacts (too small to contain recognizable faces)

                if person_roi.shape[0] < MIN_ROI_SIZE or person_roi.shape[1] < MIN_ROI_SIZE:

                    continue



                # --- STEP 2: FACE RECOGNITION (dlib via face_recognition) ---

                # Search for faces ONLY inside the person's bounding box

                # This dramatically reduces false positives and computation

                face_locations = face_recognition.face_locations(

                    person_roi, 

                    model='hog'  # Explicit HOG model (faster on CPU)

                )

                face_encodings = face_recognition.face_encodings(person_roi, face_locations)



                label = "Person (Unknown)"

                color = (0, 165, 255)  # Orange (Default - Unknown person)

                confidence = 0.0



                # Process the first face found (assume one face per person-box)

                if face_encodings:

                    face_encoding = face_encodings[0]

                    

                    # Identify the face

                    name, confidence = identify_face(

                        face_encoding, 

                        known_face_encodings, 

                        known_face_names

                    )

                    

                    if name != "Unknown":

                        color = (0, 255, 0)  # Green (Identified)

                        label = f"{name} ({confidence*100:.0f}%)"

                        

                        # Log the identification

                        log_event(name, confidence)

                    else:

                        label = "Unknown Person"



                # --- STEP 3: VISUALIZATION ---

                # Draw bounding box (YOLO coordinates)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                

                # Calculate label background size dynamically

                (label_width, label_height), _ = cv2.getTextSize(

                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2

                )

                

                # Draw label background

                cv2.rectangle(

                    frame, 

                    (x1, y1 - label_height - 10), 

                    (x1 + label_width + 10, y1), 

                    color, 

                    -1

                )

                

                # Draw label text

                cv2.putText(

                    frame, label, (x1 + 5, y1 - 5), 

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2

                )



            # Calculate smoothed FPS

            curr_time = time.time()

            fps = calculate_fps(curr_time, prev_time)

            prev_time = curr_time

            

            # Display system information overlay

            cv2.putText(

                frame, f"FPS: {fps:.1f}", (10, 30), 

                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2

            )

            cv2.putText(

                frame, f"Known: {len(known_face_names)}", (10, 70), 

                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2

            )



            # Display the frame

            cv2.imshow('Grand Brother - Edge AI', frame)



            # Check for exit key

            if cv2.waitKey(1) & 0xFF == ord('q'):

                print("[INFO] Exit signal received. Shutting down...")

                break



    except KeyboardInterrupt:

        print("\n[INFO] Interrupted by user. Shutting down...")

    except Exception as e:

        print(f"[ERROR] Unexpected error: {e}")

    finally:

        # Cleanup resources

        print("[CLEANUP] Releasing resources...")

        cap.release()

        cv2.destroyAllWindows()

        print("[DONE] System shutdown complete.")





if __name__ == "__main__":

    main()
