# detect_person_yolo.py
from ultralytics import YOLO
import cv2

# Carga modelo preentrenado (yolov8n es pequeño; cambia por yolov8s/m/l según GPU)
model = YOLO("yolov8n.pt")  # descarga automática si no existe
model.to("cpu")

# Abrir webcam (0) o archivo de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ultralytics espera imágenes en formato numpy BGR; inferencia
    results = model(frame, verbose=False)[0]  # tomamos el primer batch/result

    # Filtrar detecciones por clase 'person' (COCO class id 0)
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > 0.3:  # clase 0 = person en COCO
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("YOLO Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
