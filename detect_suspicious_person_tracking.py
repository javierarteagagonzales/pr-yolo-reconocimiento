# detect_suspicious_person_tracking.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict
import time

# =======================
# CONFIGURACIÓN DE MODELOS
# =======================

person_model = YOLO("yolov8n.pt")
person_model.to("cpu")

# Clasificadores Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# =======================
# SISTEMA DE TRACKING
# =======================

class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = 100
        
    def update(self, detections):
        current_centers = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centers.append((cx, cy, det))
        
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_info in list(self.tracks.items()):
            if track_info['frames_missing'] > 30:
                del self.tracks[track_id]
                continue
            
            last_center = track_info['center']
            best_match = None
            best_distance = self.max_distance
            
            for idx, (cx, cy, det) in enumerate(current_centers):
                if idx in matched_detections:
                    continue
                distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = idx
            
            if best_match is not None:
                cx, cy, det = current_centers[best_match]
                self.tracks[track_id]['center'] = (cx, cy)
                self.tracks[track_id]['bbox'] = det[:4]
                self.tracks[track_id]['conf'] = det[4]
                self.tracks[track_id]['frames_missing'] = 0
                self.tracks[track_id]['total_frames'] += 1
                
                # Actualizar historial de movimiento
                if 'movement_history' not in self.tracks[track_id]:
                    self.tracks[track_id]['movement_history'] = deque(maxlen=15)
                self.tracks[track_id]['movement_history'].append((cx, cy))
                
                matched_tracks.add(track_id)
                matched_detections.add(best_match)
            else:
                self.tracks[track_id]['frames_missing'] += 1
        
        for idx, (cx, cy, det) in enumerate(current_centers):
            if idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'center': (cx, cy),
                    'bbox': det[:4],
                    'conf': det[4],
                    'frames_missing': 0,
                    'total_frames': 1,
                    'first_seen': time.time(),
                    'suspicious_items': [],
                    'suspicious_score': 0,
                    'is_real_person': None,
                    'movement_history': deque(maxlen=15)
                }
                self.tracks[self.next_id]['movement_history'].append((cx, cy))
                self.next_id += 1
        
        return self.tracks

tracker = PersonTracker()

# =======================
# PARÁMETROS
# =======================

PERSON_CONFIDENCE = 0.5  # Aumentado para evitar falsos positivos
SUSPICIOUS_THRESHOLD = 2
ALERT_COOLDOWN = 5
MIN_ASPECT_RATIO = 0.3  # Relación mínima ancho/alto para persona completa
MAX_ASPECT_RATIO = 1.0  # Relación máxima ancho/alto para persona completa

suspicious_scores = defaultdict(lambda: deque(maxlen=15))
last_alert_time = {}

# =======================
# VALIDACIÓN DE PERSONA REAL
# =======================

def is_complete_person(bbox, frame_shape):
    """
    Verifica si el bounding box contiene una persona completa
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0 or height <= 0:
        return False, "Dimensiones inválidas"
    
    # Verificar proporción (persona de pie debe ser más alta que ancha)
    aspect_ratio = width / height
    
    if aspect_ratio < MIN_ASPECT_RATIO:
        return False, "Demasiado estrecho (parte del cuerpo)"
    
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, "Demasiado ancho (no es persona completa)"
    
    # Verificar tamaño mínimo (evitar detecciones muy pequeñas)
    frame_height, frame_width = frame_shape[:2]
    min_height = frame_height * 0.15  # Al menos 15% de la altura del frame
    
    if height < min_height:
        return False, "Persona muy pequeña/lejana"
    
    return True, "Persona completa detectada"

def detect_if_real_person(person_roi, movement_history):
    """
    Detecta si es una persona real o una foto/imagen
    Retorna: (is_real, confidence, reason)
    """
    confidence_score = 0
    reasons = []
    
    height, width = person_roi.shape[:2]
    
    if height < 50 or width < 30:
        return False, 0, ["ROI muy pequeño"]
    
    # 1. DETECCIÓN DE MOVIMIENTO
    if len(movement_history) >= 5:
        movements = []
        for i in range(1, len(movement_history)):
            prev_x, prev_y = movement_history[i-1]
            curr_x, curr_y = movement_history[i]
            dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(dist)
        
        avg_movement = np.mean(movements)
        
        if avg_movement > 2:  # Hay movimiento significativo
            confidence_score += 30
            reasons.append("Movimiento detectado")
        elif avg_movement < 0.5:  # Muy estático (posible foto)
            confidence_score -= 20
            reasons.append("Sin movimiento (posible foto)")
    
    # 2. DETECCIÓN DE PROFUNDIDAD (análisis de bordes)
    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Personas reales tienen más detalles que fotos planas
    if edge_density > 0.08:
        confidence_score += 25
        reasons.append("Textura rica (3D)")
    elif edge_density < 0.03:
        confidence_score -= 15
        reasons.append("Textura plana (posible foto)")
    
    # 3. DETECCIÓN DE ESTRUCTURA CORPORAL
    body_detected = upperbody_cascade.detectMultiScale(gray, 1.1, 3)
    
    if len(body_detected) > 0:
        confidence_score += 20
        reasons.append("Estructura corporal detectada")
    
    # 4. ANÁLISIS DE COLOR (fotos suelen tener colores uniformes)
    hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
    color_variance = np.std(hsv[:, :, 1])  # Varianza de saturación
    
    if color_variance > 30:
        confidence_score += 15
        reasons.append("Variación de color natural")
    elif color_variance < 15:
        confidence_score -= 10
        reasons.append("Colores uniformes (posible foto)")
    
    # 5. DETECCIÓN DE ROSTRO (personas reales tienen rostro visible frecuentemente)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    
    if len(faces) > 0:
        confidence_score += 20
        reasons.append("Rostro 3D detectado")
        
        # Verificar si los ojos se detectan (fotos 2D no suelen detectar bien ojos)
        for (fx, fy, fw, fh) in faces:
            roi_face = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 4)
            if len(eyes) >= 1:
                confidence_score += 10
                reasons.append("Ojos detectados (persona real)")
    
    # Determinar si es persona real
    is_real = confidence_score >= 50
    
    return is_real, confidence_score, reasons

# =======================
# DETECCIÓN DE ITEMS SOSPECHOSOS
# =======================

def detect_items_on_person(person_roi):
    """
    Detecta solo items realmente sospechosos
    """
    items_detected = []
    suspicious_points = 0
    
    height, width = person_roi.shape[:2]
    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    
    # 1. DETECCIÓN DE ROSTRO CUBIERTO/OCULTO
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    
    if len(faces) == 0:
        # No se detecta rostro en persona cercana = ALTAMENTE SOSPECHOSO
        if height > 150:  # Solo si la persona está lo suficientemente cerca
            items_detected.append("⚠️ Rostro completamente oculto")
            suspicious_points += 3
    else:
        # Hay rostro pero verificar si está parcialmente cubierto
        for (fx, fy, fw, fh) in faces:
            # Verificar ojos
            roi_face = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 4)
            
            if len(eyes) < 2 and height > 100:
                items_detected.append("🕶️ Lentes oscuros/Ojos ocultos")
                suspicious_points += 2
            
            # Verificar parte inferior del rostro (máscara)
            mouth_region = gray[fy+int(fh*0.55):fy+fh, fx:fx+fw]
            if mouth_region.size > 0:
                variance = cv2.Laplacian(mouth_region, cv2.CV_64F).var()
                
                if variance < 80:  # Muy baja textura en boca
                    items_detected.append("😷 Máscara/Pañuelo facial")
                    suspicious_points += 2.5
    
    # 2. DETECCIÓN DE GORRO/CAPUCHA (solo si es oscuro y cubre mucho)
    head_region = person_roi[0:int(height*0.22), :]
    if head_region.size > 0:
        hsv_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Detectar colores muy oscuros
        very_dark_mask = cv2.inRange(hsv_head, np.array([0,0,0]), np.array([180,255,60]))
        dark_percentage = np.sum(very_dark_mask > 0) / very_dark_mask.size
        
        if dark_percentage > 0.45:  # Más del 45% muy oscuro
            items_detected.append("🧢 Gorro/Capucha oscura")
            suspicious_points += 1.5
    
    # 3. DETECCIÓN DE GUANTES (manos cubiertas con material oscuro)
    hands_region = person_roi[int(height*0.65):, :]
    if hands_region.size > 0:
        hsv_hands = cv2.cvtColor(hands_region, cv2.COLOR_BGR2HSV)
        
        # Detectar negro/gris oscuro en región de manos
        glove_mask = cv2.inRange(hsv_hands, np.array([0,0,0]), np.array([180,100,70]))
        glove_percentage = np.sum(glove_mask > 0) / glove_mask.size
        
        if glove_percentage > 0.25:
            items_detected.append("🧤 Guantes oscuros")
            suspicious_points += 1
    
    # 4. DETECCIÓN DE ROPA COMPLETAMENTE OSCURA (muy sospechoso)
    hsv_full = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
    extremely_dark = cv2.inRange(hsv_full, np.array([0,0,0]), np.array([180,255,40]))
    dark_percentage = np.sum(extremely_dark > 0) / extremely_dark.size
    
    if dark_percentage > 0.60:  # Más del 60% muy oscuro
        items_detected.append("👤 Vestimenta completamente oscura")
        suspicious_points += 1
    
    return items_detected, suspicious_points

def draw_info_panel(frame, track_id, items, score, duration, bbox, is_real, real_confidence):
    """
    Dibuja panel de información con validación de persona real
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    frame_width = frame.shape[1]
    panel_width = 300
    panel_height = 180 + len(items) * 25
    
    if x2 + panel_width + 10 < frame_width:
        panel_x = x2 + 10
    else:
        panel_x = max(10, x1 - panel_width - 10)
    
    panel_y = max(10, y1)
    
    # Fondo del panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Color según estado
    if not is_real:
        color = (128, 128, 128)  # GRIS - No es persona real
        status = "NO REAL/FOTO"
    elif score >= SUSPICIOUS_THRESHOLD:
        color = (0, 0, 255)  # ROJO - Sospechoso
        status = "SOSPECHOSO"
    else:
        color = (0, 255, 0)  # VERDE - Normal
        status = "NORMAL"
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 color, 2)
    
    # Título
    cv2.putText(frame, f"ID: {track_id} - {status}", 
               (panel_x + 10, panel_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.line(frame, (panel_x + 5, panel_y + 35), 
            (panel_x + panel_width - 5, panel_y + 35), (100, 100, 100), 1)
    
    # Validación de persona real
    real_text = "Persona Real" if is_real else "Posible Foto/Imagen"
    real_color = (0, 255, 0) if is_real else (0, 165, 255)
    cv2.putText(frame, f"{real_text} ({real_confidence}%)", 
               (panel_x + 10, panel_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, real_color, 1)
    
    # Puntuación
    cv2.putText(frame, f"Puntuacion: {score:.1f}/{SUSPICIOUS_THRESHOLD}", 
               (panel_x + 10, panel_y + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    # Tiempo
    cv2.putText(frame, f"Tiempo: {duration:.1f}s", 
               (panel_x + 10, panel_y + 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Items detectados
    cv2.putText(frame, "Items sospechosos:", 
               (panel_x + 10, panel_y + 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if len(items) == 0:
        cv2.putText(frame, "  - Ninguno", 
                   (panel_x + 15, panel_y + 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    else:
        for idx, item in enumerate(items[:5]):
            cv2.putText(frame, f"  {item}", 
                       (panel_x + 15, panel_y + 165 + idx * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
    
    # Línea conectora
    cv2.line(frame, (x2, y1 + (y2-y1)//2), (panel_x, panel_y + panel_height//2), 
            color, 2)

# =======================
# CONFIGURACIÓN DE ENTRADA
# =======================

# OPCIÓN 1: Cámara (comentado)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# OPCIÓN 2: Video local
VIDEO_PATH = "test_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir el video '{VIDEO_PATH}'")
    print("\n📝 Instrucciones:")
    print("   1. Coloca un video en la misma carpeta")
    print("   2. Renombra como 'test_video.mp4' o cambia VIDEO_PATH")
    print("\nEjemplos:")
    print("   - Windows: VIDEO_PATH = 'C:/Videos/mi_video.mp4'")
    print("   - Linux/Mac: VIDEO_PATH = '/home/user/videos/mi_video.mp4'")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print("🎥 Sistema de Detección Mejorado con Validación")
print("=" * 60)
print(f"📹 Video: {VIDEO_PATH}")
print(f"⏱️  Duración: {duration:.1f}s ({total_frames} frames @ {fps} FPS)")
print("=" * 60)
print("🔍 Criterios de validación:")
print("   ✓ Diferencia persona real vs foto/imagen")
print("   ✓ Detecta solo personas completas (no partes del cuerpo)")
print("   ✓ Analiza movimiento para validar persona real")
print("\n⚠️  Items sospechosos detectables:")
print("   • Rostro completamente oculto")
print("   • Máscara/Pañuelo facial")
print("   • Lentes oscuros")
print("   • Gorro/Capucha oscura")
print("   • Guantes oscuros")
print("   • Vestimenta completamente oscura")
print("\n⌨️ Controles:")
print("   ESC - Salir | ESPACIO - Pausar | R - Reiniciar | S - Captura")
print("=" * 60)
print()

frame_count = 0
paused = False

# =======================
# BUCLE PRINCIPAL
# =======================

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\n🎬 Video finalizado. Reiniciando...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            tracker.tracks.clear()
            suspicious_scores.clear()
            last_alert_time.clear()
            continue
        
        frame_count += 1
    
    # Detectar personas
    results = person_model(frame, verbose=False)[0]
    
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > PERSON_CONFIDENCE:
            x1, y1, x2, y2 = box.tolist()
            
            # VALIDACIÓN 1: Verificar que sea persona completa
            is_complete, reason = is_complete_person((x1, y1, x2, y2), frame.shape)
            
            if not is_complete:
                continue  # Ignorar partes del cuerpo
            
            detections.append((x1, y1, x2, y2, float(conf)))
    
    # Actualizar tracker
    tracks = tracker.update(detections)
    
    suspicious_count = 0
    real_persons_count = 0
    fake_persons_count = 0
    
    for track_id, track_info in tracks.items():
        if track_info['frames_missing'] > 0:
            continue
        
        x1, y1, x2, y2 = track_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        
        # VALIDACIÓN 2: Verificar si es persona real o foto
        movement_hist = track_info.get('movement_history', deque())
        is_real, real_conf, real_reasons = detect_if_real_person(person_roi, movement_hist)
        
        track_info['is_real_person'] = is_real
        
        if not is_real:
            fake_persons_count += 1
            # Dibujar en GRIS - No es persona real
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, f"ID:{track_id} NO REAL", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            duration = time.time() - track_info['first_seen']
            draw_info_panel(frame, track_id, [], 0, duration, (x1, y1, x2, y2), 
                          False, real_conf)
            continue
        
        real_persons_count += 1
        
        # DETECCIÓN 3: Analizar items sospechosos
        items, susp_points = detect_items_on_person(person_roi)
        
        suspicious_scores[track_id].append(susp_points)
        avg_score = np.mean(suspicious_scores[track_id])
        
        track_info['suspicious_items'] = items
        track_info['suspicious_score'] = avg_score
        
        is_suspicious = avg_score >= SUSPICIOUS_THRESHOLD
        
        if is_suspicious:
            suspicious_count += 1
            color = (0, 0, 255)  # ROJO
            thickness = 3
            
            current_time = time.time()
            if (track_id not in last_alert_time or 
                current_time - last_alert_time[track_id] > ALERT_COOLDOWN):
                print(f"\n🚨 ALERTA - Persona Sospechosa ID: {track_id}")
                print(f"   Items: {', '.join(items) if items else 'Ninguno'}")
                print(f"   Puntuación: {avg_score:.1f}")
                last_alert_time[track_id] = current_time
        else:
            color = (0, 255, 0)  # VERDE
            thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"ID:{track_id} {'SOSPECHOSO' if is_suspicious else 'Normal'}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        duration = time.time() - track_info['first_seen']
        draw_info_panel(frame, track_id, items, avg_score, duration, 
                       (x1, y1, x2, y2), True, real_conf)
    
    # Panel de estadísticas
    stats_height = 180
    cv2.rectangle(frame, (10, 10), (450, stats_height), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Personas reales: {real_persons_count}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Fotos/Imagenes: {fake_persons_count}", 
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    cv2.putText(frame, f"Sospechosos: {suspicious_count}", 
               (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    cv2.putText(frame, f"Progreso: {progress:.1f}%", 
               (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    status_text = "PAUSADO" if paused else "Reproduciendo"
    status_color = (0, 165, 255) if paused else (200, 200, 200)
    cv2.putText(frame, status_text, 
               (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    cv2.imshow("Sistema de Deteccion Mejorado", frame)
    
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPACIO
        paused = not paused
        print(f"{'⏸️  Pausado' if paused else '▶️  Reanudado'}")
    elif key == ord('r') or key == ord('R'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        tracker.tracks.clear()
        suspicious_scores.clear()
        last_alert_time.clear()
        paused = False
        print("🔄 Video reiniciado")
    elif key == ord('s') or key == ord('S'):
        filename = f"captura_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Captura guardada: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Sistema finalizado")