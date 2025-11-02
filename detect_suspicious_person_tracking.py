# detect_suspicious_person_tracking.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict
import time

# =======================
# CONFIGURACIÓN DE MODELOS
# =======================

# Modelo YOLO para detectar personas
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")

# Clasificadores Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# =======================
# SISTEMA DE TRACKING
# =======================

class PersonTracker:
    def __init__(self):
        self.tracks = {}  # {track_id: info}
        self.next_id = 1
        self.max_distance = 100  # Distancia máxima para asociar detecciones
        
    def update(self, detections):
        """
        Actualiza tracks con nuevas detecciones
        detections: lista de (x1, y1, x2, y2, conf)
        """
        current_centers = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centers.append((cx, cy, det))
        
        # Asociar detecciones con tracks existentes
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_info in list(self.tracks.items()):
            if track_info['frames_missing'] > 30:  # Eliminar tracks perdidos
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
                # Actualizar track existente
                cx, cy, det = current_centers[best_match]
                self.tracks[track_id]['center'] = (cx, cy)
                self.tracks[track_id]['bbox'] = det[:4]
                self.tracks[track_id]['conf'] = det[4]
                self.tracks[track_id]['frames_missing'] = 0
                self.tracks[track_id]['total_frames'] += 1
                matched_tracks.add(track_id)
                matched_detections.add(best_match)
            else:
                # Incrementar frames perdidos
                self.tracks[track_id]['frames_missing'] += 1
        
        # Crear nuevos tracks para detecciones no asociadas
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
                    'suspicious_score': 0
                }
                self.next_id += 1
        
        return self.tracks

tracker = PersonTracker()

# =======================
# PARÁMETROS DE DETECCIÓN
# =======================

PERSON_CONFIDENCE = 0.3
SUSPICIOUS_THRESHOLD = 2
ALERT_COOLDOWN = 5

suspicious_scores = defaultdict(lambda: deque(maxlen=15))
last_alert_time = {}

# =======================
# FUNCIONES DE DETECCIÓN DE PRENDAS/OBJETOS
# =======================

def detect_items_on_person(person_roi):
    """
    Detecta prendas y objetos específicos en la persona
    Retorna: lista de items detectados con descripción
    """
    items_detected = []
    suspicious_points = 0
    
    height, width = person_roi.shape[:2]
    
    # 1. DETECCIÓN DE GORRO/CAPUCHA
    head_region = person_roi[0:int(height*0.25), :]
    if head_region.size > 0:
        hsv_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Detectar materiales oscuros (gorros)
        dark_mask = cv2.inRange(hsv_head, np.array([0,0,0]), np.array([180,255,80]))
        dark_percentage = np.sum(dark_mask > 0) / dark_mask.size
        
        if dark_percentage > 0.35:
            items_detected.append("Gorro/Capucha oscura")
            suspicious_points += 1
    
    # 2. DETECCIÓN DE MÁSCARA FACIAL
    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            # Región de boca/nariz
            mouth_region = gray[fy+int(fh*0.5):fy+fh, fx:fx+fw]
            
            if mouth_region.size > 0:
                # Detectar cobertura por baja varianza
                variance = cv2.Laplacian(mouth_region, cv2.CV_64F).var()
                
                if variance < 100:
                    items_detected.append("Mascara/Pañuelo facial")
                    suspicious_points += 2
                    
                # Detectar color de máscara
                mouth_color = person_roi[fy+int(fh*0.5):fy+fh, fx:fx+fw]
                if mouth_color.size > 0:
                    hsv_mouth = cv2.cvtColor(mouth_color, cv2.COLOR_BGR2HSV)
                    # Máscaras médicas (blancas/azules)
                    white_mask = cv2.inRange(hsv_mouth, np.array([0,0,180]), np.array([180,30,255]))
                    if np.sum(white_mask > 0) / white_mask.size > 0.3:
                        if "Mascara/Pañuelo facial" not in items_detected:
                            items_detected.append("Mascara medica")
                            suspicious_points += 1
    else:
        # No se detecta rostro = muy sospechoso
        items_detected.append("Rostro completamente cubierto")
        suspicious_points += 3
    
    # 3. DETECCIÓN DE LENTES OSCUROS
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi_gray_face = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray_face, 1.1, 4)
            
            if len(eyes) < 2:
                items_detected.append("Lentes oscuros/Gafas")
                suspicious_points += 2
    
    # 4. DETECCIÓN DE BUFANDA/CUELLO ALTO
    neck_region = person_roi[int(height*0.2):int(height*0.4), int(width*0.3):int(width*0.7)]
    if neck_region.size > 0:
        hsv_neck = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)
        # Detectar tejidos (bufandas suelen tener texturas repetitivas)
        gray_neck = cv2.cvtColor(neck_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_neck, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:
            items_detected.append("Bufanda/Cuello alto")
            suspicious_points += 1
    
    # 5. DETECCIÓN DE ROPA OSCURA (sospechosa en ciertos contextos)
    torso_region = person_roi[int(height*0.3):int(height*0.7), :]
    if torso_region.size > 0:
        hsv_torso = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        very_dark = cv2.inRange(hsv_torso, np.array([0,0,0]), np.array([180,255,50]))
        dark_percentage = np.sum(very_dark > 0) / very_dark.size
        
        if dark_percentage > 0.5:
            items_detected.append("Ropa muy oscura")
            suspicious_points += 0.5
    
    # 6. DETECCIÓN DE GUANTES
    hands_region = person_roi[int(height*0.6):, :]
    if hands_region.size > 0:
        hsv_hands = cv2.cvtColor(hands_region, cv2.COLOR_BGR2HSV)
        # Detectar negro/oscuro en manos
        glove_mask = cv2.inRange(hsv_hands, np.array([0,0,0]), np.array([180,255,60]))
        glove_percentage = np.sum(glove_mask > 0) / glove_mask.size
        
        if glove_percentage > 0.2:
            items_detected.append("Guantes oscuros")
            suspicious_points += 1
    
    return items_detected, suspicious_points

def draw_info_panel(frame, track_id, items, score, duration, bbox):
    """
    Dibuja panel de información junto a la persona detectada
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Determinar posición del panel (derecha o izquierda según espacio)
    frame_width = frame.shape[1]
    panel_width = 280
    panel_height = 150 + len(items) * 25
    
    if x2 + panel_width + 10 < frame_width:
        # Panel a la derecha
        panel_x = x2 + 10
    else:
        # Panel a la izquierda
        panel_x = max(10, x1 - panel_width - 10)
    
    panel_y = max(10, y1)
    
    # Fondo del panel con transparencia
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Borde del panel
    color = (0, 0, 255) if score >= SUSPICIOUS_THRESHOLD else (0, 255, 0)
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 color, 2)
    
    # Título
    status = "SOSPECHOSO" if score >= SUSPICIOUS_THRESHOLD else "NORMAL"
    cv2.putText(frame, f"ID: {track_id} - {status}", 
               (panel_x + 10, panel_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Línea separadora
    cv2.line(frame, (panel_x + 5, panel_y + 35), 
            (panel_x + panel_width - 5, panel_y + 35), (100, 100, 100), 1)
    
    # Puntuación
    cv2.putText(frame, f"Puntuacion: {score:.1f}/{SUSPICIOUS_THRESHOLD}", 
               (panel_x + 10, panel_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    # Tiempo de tracking
    cv2.putText(frame, f"Tiempo: {duration:.1f}s", 
               (panel_x + 10, panel_y + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Items detectados
    cv2.putText(frame, "Items detectados:", 
               (panel_x + 10, panel_y + 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if len(items) == 0:
        cv2.putText(frame, "  - Ninguno", 
                   (panel_x + 15, panel_y + 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    else:
        for idx, item in enumerate(items[:5]):  # Max 5 items
            cv2.putText(frame, f"  - {item}", 
                       (panel_x + 15, panel_y + 135 + idx * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
    
    # Línea conectando panel con persona
    cv2.line(frame, (x2, y1 + (y2-y1)//2), (panel_x, panel_y + panel_height//2), 
            color, 2)

# =======================
# CONFIGURACIÓN DE ENTRADA DE VIDEO
# =======================

# OPCIÓN 1: Usar cámara web (comentado para pruebas)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# OPCIÓN 2: Usar video local para pruebas
VIDEO_PATH = "test_video.mp4"  # Cambia esto por la ruta de tu video
cap = cv2.VideoCapture(VIDEO_PATH)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir el video '{VIDEO_PATH}'")
    print("\n📝 Instrucciones:")
    print("   1. Coloca un video en la misma carpeta que este script")
    print("   2. Renombra el video como 'test_video.mp4' o")
    print("   3. Cambia la variable VIDEO_PATH en la línea 253 con la ruta correcta")
    print("\nEjemplos de rutas:")
    print("   - Windows: VIDEO_PATH = 'C:/Videos/mi_video.mp4'")
    print("   - Linux/Mac: VIDEO_PATH = '/home/user/videos/mi_video.mp4'")
    print("   - Relativa: VIDEO_PATH = 'videos/test.mp4'")
    exit()

# Obtener información del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print("🎥 Sistema de Detección con Tracking Activado")
print("=" * 50)
print(f"📹 Video: {VIDEO_PATH}")
print(f"⏱️  Duración: {duration:.1f} segundos ({total_frames} frames a {fps} FPS)")
print("=" * 50)
print("📋 Items detectables:")
print("   • Gorro/Capucha oscura")
print("   • Máscara/Pañuelo facial")
print("   • Lentes oscuros")
print("   • Bufanda/Cuello alto")
print("   • Ropa muy oscura")
print("   • Guantes oscuros")
print("   • Rostro completamente cubierto")
print("\n⌨️ Controles:")
print("   ESC - Salir")
print("   ESPACIO - Pausar/Reanudar")
print("   R - Reiniciar video")
print("   S - Guardar captura actual")
print("=" * 50)
print()

frame_count = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\n🎬 Video finalizado. Reiniciando...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video
            frame_count = 0
            tracker.tracks.clear()
            suspicious_scores.clear()
            last_alert_time.clear()
            continue
        
        frame_count += 1
    
    # Detectar personas
    results = person_model(frame, verbose=False)[0]
    
    # Preparar detecciones para el tracker
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > PERSON_CONFIDENCE:
            x1, y1, x2, y2 = box.tolist()
            detections.append((x1, y1, x2, y2, float(conf)))
    
    # Actualizar tracker
    tracks = tracker.update(detections)
    
    # Procesar cada track
    suspicious_count = 0
    
    for track_id, track_info in tracks.items():
        if track_info['frames_missing'] > 0:
            continue  # Saltar tracks no visibles
        
        x1, y1, x2, y2 = track_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extraer ROI
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        
        # Detectar items y calcular puntuación
        items, susp_points = detect_items_on_person(person_roi)
        
        # Actualizar historial de puntuación
        suspicious_scores[track_id].append(susp_points)
        avg_score = np.mean(suspicious_scores[track_id])
        
        # Guardar en track
        track_info['suspicious_items'] = items
        track_info['suspicious_score'] = avg_score
        
        # Determinar si es sospechoso
        is_suspicious = avg_score >= SUSPICIOUS_THRESHOLD
        
        if is_suspicious:
            suspicious_count += 1
            color = (0, 0, 255)  # ROJO
            thickness = 3
            
            # Alerta con cooldown
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
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label principal
        label = f"ID:{track_id} {'SOSPECHOSO' if is_suspicious else 'Normal'}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Duración de tracking
        duration = time.time() - track_info['first_seen']
        
        # Dibujar panel de información
        draw_info_panel(frame, track_id, items, avg_score, duration, (x1, y1, x2, y2))
    
    # Panel de estadísticas generales
    stats_height = 150
    cv2.rectangle(frame, (10, 10), (400, stats_height), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Personas rastreadas: {len([t for t in tracks.values() if t['frames_missing'] == 0])}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Sospechosos activos: {suspicious_count}", 
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Progreso del video
    progress_percentage = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    cv2.putText(frame, f"Progreso: {progress_percentage:.1f}%", 
               (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    # Estado de pausa
    status_text = "PAUSADO" if paused else "Reproduciendo"
    status_color = (0, 165, 255) if paused else (200, 200, 200)
    cv2.putText(frame, status_text, 
               (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    cv2.imshow("Sistema de Deteccion con Tracking", frame)
    
    # Controles de teclado
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    
    if key == 27:  # ESC - Salir
        break
    elif key == 32:  # ESPACIO - Pausar/Reanudar
        paused = not paused
        print(f"{'⏸️  Pausado' if paused else '▶️  Reanudado'}")
    elif key == ord('r') or key == ord('R'):  # R - Reiniciar
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        tracker.tracks.clear()
        suspicious_scores.clear()
        last_alert_time.clear()
        paused = False
        print("🔄 Video reiniciado")
    elif key == ord('s') or key == ord('S'):  # S - Guardar captura
        filename = f"captura_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Captura guardada: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Sistema finalizado")