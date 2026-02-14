
#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
import os

def main():
    # 1. Cargar el Modelo
    # La primera vez descargará 'yolov8n.pt' (6MB) automáticamente
    print("[INIT] Cargando modelo YOLOv8 Nano... (esto puede tardar la primera vez)")
    model = YOLO('yolov8n.onnx')

    # 2. Configurar Cámara
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[RUN] Iniciando Detección de Personas. Presiona 'q' para salir.")

    frame_count = 0
    start_time = time.time()
    logs = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. INFERENCIA (La magia de la IA)
        # classes=[0] -> Solo detecta personas (ignora sillas, botellas, etc)
        # conf=0.4 -> Solo muestra si está 40% seguro
        # verbose=False -> Para no llenar la terminal de texto
        results = model.predict(frame, conf=0.4, classes=[0], verbose=False)

        # 4. Procesar Resultados
        # YOLO nos devuelve un objeto 'Results', podemos pintarlo directo:
        annotated_frame = results[0].plot()

        # Extraer datos para tu Tesis
        # results[0].boxes.cls son las clases detectadas
        person_count = len(results[0].boxes)

        if person_count > 0:
            logs.append({
                'timestamp': datetime.now(),
                'person_count': person_count,
                'max_conf': float(results[0].boxes.conf.max()) # Confianza más alta
            })

        # 5. Calcular FPS (Rendimiento Real)
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        # Dibujar FPS en pantalla
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Mostrar
        cv2.imshow('Grand Brother - YOLOv8 (AI)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Guardar Datos
    if logs:
        df = pd.DataFrame(logs)
        os.makedirs('../logs', exist_ok=True)
        filename = f"../logs/yolo_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        print(f"[INFO] Datos guardados en {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
