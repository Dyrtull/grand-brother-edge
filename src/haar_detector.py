#!/usr/bin/env python3
import cv2
import time
import pandas as pd
from datetime import datetime
import os

def main():
    # Cargar el modelo pre-entrenado de OpenCV (Rostros frontales)
    # POR ESTO (Ruta absoluta del sistema):
    cascade_path = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

    # Opcional: Si quieres ser a prueba de balas, agrega esto para validar:

    if not os.path.exists(cascade_path):
    	print(f"[ERROR] No encuentro el archivo en: {cascade_path}")
    	exit()



    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0) # Ajusta a 0 o 1 según te funcionó antes
    cap.set(3, 640) # Ancho
    cap.set(4, 480) # Alto

    print("[INFO] Iniciando Haar Detector. Presiona 'q' para salir.")

    frame_count = 0
    start_time = time.time()
    logs = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Haar funciona mejor en escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # DETECCIÓN
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Dibujar rectángulos
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Guardar datos para tesis (si hay rostros)
        if len(faces) > 0:
            logs.append({'timestamp': datetime.now(), 'count': len(faces)})

        # FPS
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Grand Brother - Haar', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar CSV al salir
    if logs:
        df = pd.DataFrame(logs)
        os.makedirs('../logs', exist_ok=True)
        df.to_csv(f'../logs/haar_{int(time.time())}.csv', index=False)
        print("[INFO] Logs guardados.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
