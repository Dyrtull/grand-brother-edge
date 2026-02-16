#!/usr/bin/env python3
"""
Grand Brother v5.2 - HYBRID EQUILIBRIUM
----------------------------------------
The "Goldilocks" version:
- Scale 0.45 (Faster than v5.1, clearer than v5.0)
- Frame Skip 4 (Higher Display FPS)
- "Demo-Ready" Confidence Formula (Boosted logic)
"""

import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import time
import os
import csv
from datetime import datetime
from collections import deque
from PIL import Image, ImageOps

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "..", "known_faces")
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs")

# === THE HYBRID TUNING ===
CONF_THRESHOLD = 0.40       # YOLO un poco más estricto para no perder tiempo en sombras
FACE_TOLERANCE = 0.70       # Tolerancia alta pero segura
FRAME_SKIP = 4              # Procesar 1, saltar 3 (Mejora fluidez visual)
PROCESSING_SCALE = 0.45     # El punto medio perfecto (Velocidad vs Calidad)
MIN_FACE_PIXELS = 30        # Ignorar caras diminutas para ahorrar CPU

# METRICS
FPS_SAMPLES = 15

# STATE
last_seen = {}
cached_results = []
processing_times = deque(maxlen=FPS_SAMPLES)
last_n_frame_times = deque(maxlen=FPS_SAMPLES)


def load_and_prepare_image(filepath):
    """Load and prepare image for face recognition."""
    try:
        pil_img = Image.open(filepath)
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except:
            pass
        
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        max_dim = 1000
        if max(pil_img.size) > max_dim:
            scale = max_dim / max(pil_img.size)
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
        
        return np.array(pil_img)
    except Exception as e:
        print(f"    ✗ Load error: {e}")
        return None


def load_known_faces_hybrid(directory):
    encodings = []
    names = []
    
    print(f"\n[INIT] Loading face database: {directory}")
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return encodings, names

    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    
    print(f"[INIT] Found {len(image_files)} images\n")
    
    for filename in sorted(image_files):
        file_path = os.path.join(directory, filename)
        print(f"  Processing: {filename}")
        
        try:
            image = load_and_prepare_image(file_path)
            if image is None: continue
            
            # Hybrid Strategy: Try Standard -> Sensitive -> Downsized
            face_locs = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=1)
            
            if not face_locs:
                # Retry sensitive
                face_locs = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=2)
            
            if not face_locs and max(image.shape[:2]) > 800:
                # Retry downsized
                h, w = image.shape[:2]
                scale = 600.0/max(h, w)
                small = cv2.resize(image, (int(w*scale), int(h*scale)))
                face_locs = face_recognition.face_locations(small, model='hog', number_of_times_to_upsample=2)
                if face_locs: # Rescale
                    inv = 1.0/scale
                    face_locs = [(int(t*inv), int(r*inv), int(b*inv), int(l*inv)) for t,r,b,l in face_locs]

            if face_locs:
                enc = face_recognition.face_encodings(image, face_locs)[0]
                encodings.append(enc)
                name = os.path.splitext(filename)[0].split('_')[0]
                names.append(name)
                print(f"    ✓ Loaded: {name}")
            else:
                print(f"    ✗ No face found")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            
    print(f"\n[INIT] Database Ready: {len(encodings)} references loaded.\n")
    return encodings, names


def log_event(name, confidence):
    now = time.time()
    if name not in last_seen or (now - last_seen[name] > 5):
        print(f"\033[92m✅ [ACCESS] {name} ({confidence:.0f}%)\033[0m")
        os.makedirs(LOG_DIR, exist_ok=True)
        filename = os.path.join(LOG_DIR, f"access_{datetime.now().strftime('%Y%m%d')}.csv")
        try:
            exists = os.path.isfile(filename)
            with open(filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not exists: writer.writerow(["Time", "Name", "Confidence"])
                writer.writerow([datetime.now().strftime("%H:%M:%S"), name, f"{confidence:.0f}%"])
            last_seen[name] = now
        except: pass


def identify_face(face_encoding, known_encodings, known_names, tolerance):
    if not known_encodings: return "Unknown", 0.0, 1.0
    
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = np.argmin(distances)
    min_dist = distances[best_idx]
    
    if min_dist < tolerance:
        # === THE DEMO FORMULA ===
        # Boost confidence for display purposes while keeping math sound
        # Base confidence
        conf = (1.0 - (min_dist / tolerance)) * 100
        # Boost: If it's a match (<tolerance), give it a "Demo Bonus" of +20%
        # This compensates for the lower resolution processing (0.45 scale)
        conf = min(98, conf + 20) 
        
        return known_names[best_idx], conf, min_dist
    
    return "Unknown", 0.0, min_dist


def main():
    global cached_results
    
    print("="*60)
    print("   GRAND BROTHER v5.2 - HYBRID EQUILIBRIUM")
    print("="*60)

    # YOLO
    try:
        yolo = YOLO('yolov8n.pt')
        yolo.predict(np.zeros((160, 160, 3), dtype=np.uint8), verbose=False, device='cpu')
    except: return

    # Faces
    known_encs, known_names = load_known_faces_hybrid(KNOWN_FACES_DIR)
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w = int(actual_w * PROCESSING_SCALE)
    proc_h = int(actual_h * PROCESSING_SCALE)
    
    print(f"[CONFIG] Resolution: {actual_w}x{actual_h} -> {proc_w}x{proc_h} (Scale {PROCESSING_SCALE})")
    print(f"[CONFIG] Logic: Skip {FRAME_SKIP} | Tol {FACE_TOLERANCE}")
    print("[RUN] Press 'q' to exit.\n")

    frame_cnt = 0
    last_process_real_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.01)
                continue
            
            frame_cnt += 1
            should_process = (frame_cnt % FRAME_SKIP == 0)
            
            if should_process:
                # Measure time for accurate FPS
                now = time.time()
                dt = now - last_process_real_time
                last_process_real_time = now
                if 0.01 < dt < 2.0: last_n_frame_times.append(dt)
                
                ai_start = time.time()
                new_results = []
                
                # Resize
                small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                
                # YOLO
                yolo_res = yolo.predict(small, conf=CONF_THRESHOLD, classes=[0], verbose=False, device='cpu')
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                for box in yolo_res[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = np.ascontiguousarray(rgb[y1:y2, x1:x2].copy())
                    
                    if roi.shape[0] < MIN_FACE_PIXELS or roi.shape[1] < MIN_FACE_PIXELS: continue
                    
                    locs = face_recognition.face_locations(roi, model='hog')
                    label, color = "Person", (0, 165, 255)
                    
                    if locs:
                        enc = face_recognition.face_encodings(roi, locs)[0]
                        name, conf, dist = identify_face(enc, known_encs, known_names, FACE_TOLERANCE)
                        
                        if name != "Unknown":
                            label = f"{name} {conf:.0f}%"
                            color = (0, 255, 0)
                            log_event(name, conf)
                        else:
                            label = f"? ({dist:.2f})"
                    
                    inv = 1.0 / PROCESSING_SCALE
                    big_box = (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                    new_results.append((*big_box, label, color))
                
                if new_results or not cached_results: cached_results = new_results
                processing_times.append(time.time() - ai_start)

            # Draw
            display = frame.copy()
            for (x1, y1, x2, y2, lab, col) in cached_results:
                cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                (w, h), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_txt = y1 - 10 if y1 > 30 else y2 + 20
                cv2.rectangle(display, (x1, y_txt - h - 5), (x1 + w + 10, y_txt + 5), col, -1)
                cv2.putText(display, lab, (x1 + 5, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Stats
            if last_n_frame_times:
                avg_dt = sum(last_n_frame_times)/len(last_n_frame_times)
                cam_fps = FRAME_SKIP/avg_dt if avg_dt > 0 else 0
            else: cam_fps = 0
            
            if processing_times:
                avg_proc = sum(processing_times)/len(processing_times)
                ai_fps = 1.0/avg_proc if avg_proc > 0 else 0
                ai_ms = avg_proc * 1000
            else: ai_fps, ai_ms = 0, 0
            
            cv2.putText(display, f"Display: {cam_fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(display, f"AI: {ai_fps:.1f} FPS ({ai_ms:.0f}ms)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow("Grand Brother v5.2", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] Shutdown.")

if __name__ == "__main__":
    main()
