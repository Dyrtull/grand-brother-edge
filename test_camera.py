#!/usr/bin/env python3
import cv2
import sys

def test_camera(camera_index=0):
    print(f"[INFO] Abriendo cámara {camera_index}...")

    # Intenta abrir la cámara
    # Si usas la cámara oficial de Pi (Ribbon cable), a veces es index 0, o usamos libcamera
    # Si es USB, suele ser 0 o 1.
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
        print("Intenta cambiar el índice a 1 o verifica conexión.")
        return

    # Configura resolución básica para asegurar fluidez
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Cámara iniciada. Presiona 'q' en la ventana de video para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo recibir frame (stream end?). Saliendo...")
            break

        # Dibuja un círculo verde en el centro para confirmar que podemos editar la imagen
        height, width = frame.shape[:2]
        cv2.circle(frame, (width//2, height//2), 20, (0, 255, 0), 2)

        cv2.imshow('Grand Brother - Test Camara', frame)

        # Salir con tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test finalizado.")

if __name__ == "__main__":
    # Si tienes la cámara USB y la oficial conectadas, prueba cambiar 0 por 1 si no sale nada
    test_camera(0)
