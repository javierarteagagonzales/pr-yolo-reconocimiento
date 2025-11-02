# detect_suspicious_person.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time

# =======================
# CONFIGURACIÓN DE MODELOS
# =======================

# Modelo YOLO para detectar personas
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")

# Modelo YOLO para detectar accesorios (gorro, lentes, máscaras, etc.)
# Usaremos YOLOv8 entrenado en más clases o un modelo personalizado
try:
    # Intenta cargar modelo con más clases (incluye accesorios)
    accessory_model = YOLO("yolov8n.pt")  # Usaremos el mismo por ahora
    accessory_model.to("cpu")
except:
    print("⚠️ Modelo de accesorios no disponible")
    accessory_model = None

# Clasificador Haar Cascade para detección de rostros (alternativa ligera)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# =======================
# PARÁMETROS DE DETECCIÓN
# =======================

PERSON_CONFIDENCE = 0.3  # Confianza mínima para detectar persona
SUSPICIOUS_THRESHOLD = 2  # Puntos necesarios para marcar como sospechoso
ALERT_COOLDOWN = 3  # Segundos entre alertas del mismo sospechoso

# Sistema de puntuación de sospecha
suspicious_scores = {}
last_alert_time = {}

# =======================
# FUNCIONES DE ANÁLISIS
# =======================

def analyze_face_coverage(person_roi):
    """
    Analiza si el rostro está visible o cubierto
    Retorna: puntos de sospecha (0-3)
    """
    suspicious_points = 0
    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostro
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    
    if len(faces) == 0:
        # No se detecta rostro = MUY SOSPECHOSO
        suspicious_points += 3
        return suspicious_points, "Sin rostro visible"
    
    # Si hay rostro, verificar si los ojos están visibles
    for (x, y, w, h) in faces:
        roi_gray_face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray_face, 1.1, 4)
        
        if len(eyes) < 2:
            # Ojos no detectados = posible uso de lentes oscuros o máscara
            suspicious_points += 2
            return suspicious_points, "Ojos no visibles"
    
    return suspicious_points, "Rostro visible"

def detect_head_coverage(person_roi):
    """
    Detecta si la persona lleva algo en la cabeza (gorro, capucha)
    Basado en análisis de color y forma en la parte superior
    """
    suspicious_points = 0
    height, width = person_roi.shape[:2]
    
    # Analizar tercio superior (donde estaría la cabeza)
    head_region = person_roi[0:int(height*0.3), :]
    
    # Convertir a HSV para mejor detección de colores oscuros
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    
    # Detectar colores oscuros (común en gorros/capuchas)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    dark_percentage = np.sum(dark_mask > 0) / dark_mask.size
    
    if dark_percentage > 0.4:  # Más del 40% oscuro en la cabeza
        suspicious_points += 1
        return suspicious_points, "Posible gorro/capucha"
    
    return suspicious_points, "Cabeza descubierta"

def detect_face_mask(person_roi):
    """
    Detecta si hay una máscara o cobertura facial
    Analiza la región de la boca y nariz
    """
    suspicious_points = 0
    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Región de boca/nariz (centro-inferior del rostro)
    mouth_region = gray[int(height*0.5):int(height*0.8), int(width*0.3):int(width*0.7)]
    
    if mouth_region.size == 0:
        return 0, "No analizable"
    
    # Calcular varianza - rostro cubierto tiene menos textura
    variance = cv2.Laplacian(mouth_region, cv2.CV_64F).var()
    
    if variance < 50:  # Poca textura = posible cobertura
        suspicious_points += 2
        return suspicious_points, "Posible máscara/cobertura"
    
    return suspicious_points, "Boca/nariz visible"

def calculate_suspicious_score(person_id, person_roi):
    """
    Calcula puntuación total de sospecha
    """
    total_score = 0
    reasons = []
    
    # Análisis 1: Cobertura facial
    score1, reason1 = analyze_face_coverage(person_roi)
    total_score += score1
    if score1 > 0:
        reasons.append(reason1)
    
    # Análisis 2: Cobertura de cabeza
    score2, reason2 = detect_head_coverage(person_roi)
    total_score += score2
    if score2 > 0:
        reasons.append(reason2)
    
    # Análisis 3: Máscara facial
    score3, reason3 = detect_face_mask(person_roi)
    total_score += score3
    if score3 > 0:
        reasons.append(reason3)
    
    # Actualizar historial de puntuaciones
    if person_id not in suspicious_scores:
        suspicious_scores[person_id] = deque(maxlen=10)
    suspicious_scores[person_id].append(total_score)
    
    # Promedio de últimas detecciones para estabilidad
    avg_score = np.mean(suspicious_scores[person_id])
    
    return avg_score, reasons

# =======================
# BUCLE PRINCIPAL
# =======================

cap = cv2.VideoCapture(0)

# Configurar resolución (opcional, mejora rendimiento)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Iniciando detección de personas sospechosas...")
print("📋 Criterios de sospecha:")
print("   - Rostro no visible")
print("   - Uso de gorro/capucha")
print("   - Uso de máscara o cobertura facial")
print("   - Lentes oscuros (ojos no detectables)")
print("\n⌨️ Presiona ESC para salir\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Detectar personas cada frame
    results = person_model(frame, verbose=False)[0]
    
    # Contadores de pantalla
    total_persons = 0
    suspicious_count = 0
    
    # Procesar cada detección de persona
    for idx, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, 
                                                 results.boxes.cls, 
                                                 results.boxes.conf)):
        if int(cls) == 0 and conf > PERSON_CONFIDENCE:  # Clase 0 = persona
            total_persons += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Extraer ROI de la persona
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Calcular puntuación de sospecha
            person_id = f"p{idx}"
            susp_score, reasons = calculate_suspicious_score(person_id, person_roi)
            
            # Determinar si es sospechoso
            is_suspicious = susp_score >= SUSPICIOUS_THRESHOLD
            
            if is_suspicious:
                suspicious_count += 1
                # Color ROJO para sospechoso
                color = (0, 0, 255)
                thickness = 3
                label = f"SOSPECHOSO {conf:.2f}"
                
                # Sistema de alertas con cooldown
                current_time = time.time()
                if (person_id not in last_alert_time or 
                    current_time - last_alert_time[person_id] > ALERT_COOLDOWN):
                    print(f"🚨 ALERTA: Persona sospechosa detectada!")
                    print(f"   Razones: {', '.join(reasons)}")
                    print(f"   Puntuación: {susp_score:.1f}/{SUSPICIOUS_THRESHOLD}")
                    last_alert_time[person_id] = current_time
                
                # Mostrar razones en pantalla
                y_offset = y1 - 30
                for reason in reasons[:3]:  # Máximo 3 razones
                    cv2.putText(frame, reason, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    y_offset -= 15
                
            else:
                # Color VERDE para persona normal
                color = (0, 255, 0)
                thickness = 2
                label = f"Persona {conf:.2f}"
            
            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Mostrar puntuación
            score_text = f"Score: {susp_score:.1f}"
            cv2.putText(frame, score_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Panel de información en pantalla
    cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {total_persons}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Sospechosos: {suspicious_count}", 
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "ESC para salir", 
               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Mostrar frame
    cv2.imshow("Deteccion de Personas Sospechosas", frame)
    
    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Sistema de detección finalizado")