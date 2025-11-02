# tracking_marked_suspicious.py
"""
Script para hacer tracking especial de personas marcadas como sospechosas
desde la interfaz web
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import time
from collections import deque

# =======================
# CONFIGURACIÓN
# =======================

DATA_FILE = "persons_data.json"
VIDEO_PATH = "test_video.mp4"
ALERTS_LOG = "alerts_log.txt"

# Modelo
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")

# =======================
# RECONOCIMIENTO FACIAL SIMPLE
# =======================

def load_suspicious_persons_features():
    """
    Carga características de personas sospechosas marcadas
    """
    if not os.path.exists(DATA_FILE):
        return {}
    
    with open(DATA_FILE, 'r') as f:
        persons_db = json.load(f)
    
    suspicious_features = {}
    
    for person_id, data in persons_db.items():
        if data.get('is_suspicious', False):
            # Cargar imagen
            image_path = data['image_path']
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    # Extraer características simples (histograma de color)
                    hist_features = extract_color_histogram(img)
                    suspicious_features[person_id] = {
                        'features': hist_features,
                        'name': f"Sospechoso #{data['track_id']}",
                        'marked_date': data['date']
                    }
    
    return suspicious_features

def extract_color_histogram(image):
    """
    Extrae histograma de color como característica simple
    """
    # Convertir a HSV para mejor comparación
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calcular histogramas
    h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])
    
    # Normalizar
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    return np.concatenate([h_hist, s_hist, v_hist])

def compare_persons(features1, features2):
    """
    Compara dos personas usando sus características
    Retorna: similitud (0-100)
    """
    # Correlación entre histogramas
    correlation = cv2.compareHist(
        features1.astype(np.float32).reshape(-1, 1),
        features2.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_CORREL
    )
    
    # Convertir a porcentaje
    similarity = max(0, correlation * 100)
    
    return similarity

# =======================
# SISTEMA DE ALERTAS
# =======================

def log_alert(message):
    """
    Registra alerta en archivo de log
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(ALERTS_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def show_alert_overlay(frame, message, duration=3):
    """
    Muestra alerta en pantalla con overlay
    """
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Fondo rojo semi-transparente
    cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Texto de alerta
    cv2.putText(frame, "⚠️ ALERTA DE SEGURIDAD ⚠️", 
               (width//2 - 300, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.putText(frame, message, 
               (width//2 - 250, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# =======================
# TRACKING CON RECONOCIMIENTO
# =======================

class SuspiciousPersonTracker:
    def __init__(self, suspicious_features):
        self.suspicious_features = suspicious_features
        self.tracks = {}
        self.next_id = 1
        self.max_distance = 100
        self.alerts_sent = {}  # Cooldown de alertas
        self.alert_cooldown = 5  # segundos
        
    def update(self, detections, frame):
        current_centers = []
        detected_matches = []
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Extraer ROI
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Buscar coincidencias con personas sospechosas
            match_id = None
            max_similarity = 0
            
            if person_roi.size > 0:
                person_features = extract_color_histogram(person_roi)
                
                for susp_id, susp_data in self.suspicious_features.items():
                    similarity = compare_persons(person_features, susp_data['features'])
                    
                    if similarity > 65 and similarity > max_similarity:  # Umbral de 65%
                        max_similarity = similarity
                        match_id = susp_id
            
            current_centers.append((cx, cy, det, match_id, max_similarity))
        
        matched_detections = set()
        
        # Actualizar tracks existentes
        for track_id, track_info in list(self.tracks.items()):
            if track_info['frames_missing'] > 30:
                del self.tracks[track_id]
                continue
            
            last_center = track_info['center']
            best_match = None
            best_distance = self.max_distance
            
            for idx, (cx, cy, det, match_id, similarity) in enumerate(current_centers):
                if idx in matched_detections:
                    continue
                distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = idx
            
            if best_match is not None:
                cx, cy, det, match_id, similarity = current_centers[best_match]
                self.tracks[track_id]['center'] = (cx, cy)
                self.tracks[track_id]['bbox'] = det[:4]
                self.tracks[track_id]['conf'] = det[4]
                self.tracks[track_id]['frames_missing'] = 0
                self.tracks[track_id]['total_frames'] += 1
                
                # Actualizar match si encontró uno mejor
                if match_id and similarity > track_info.get('best_match_similarity', 0):
                    track_info['matched_suspicious_id'] = match_id
                    track_info['best_match_similarity'] = similarity
                
                matched_detections.add(best_match)
            else:
                self.tracks[track_id]['frames_missing'] += 1
        
        # Crear nuevos tracks
        for idx, (cx, cy, det, match_id, similarity) in enumerate(current_centers):
            if idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'center': (cx, cy),
                    'bbox': det[:4],
                    'conf': det[4],
                    'frames_missing': 0,
                    'total_frames': 1,
                    'first_seen': time.time(),
                    'matched_suspicious_id': match_id,
                    'best_match_similarity': similarity if match_id else 0,
                    'alert_shown': False
                }
                
                # Enviar alerta si es match con sospechoso
                if match_id:
                    self.send_alert(self.next_id, match_id, similarity)
                
                self.next_id += 1
        
        return self.tracks
    
    def send_alert(self, track_id, suspicious_id, similarity):
        """
        Envía alerta cuando se detecta persona sospechosa
        """
        current_time = time.time()
        
        # Verificar cooldown
        if suspicious_id in self.alerts_sent:
            if current_time - self.alerts_sent[suspicious_id] < self.alert_cooldown:
                return
        
        susp_name = self.suspicious_features[suspicious_id]['name']
        marked_date = self.suspicious_features[suspicious_id]['marked_date']
        
        alert_msg = f"🚨 DETECTADO: {susp_name} (Similitud: {similarity:.0f}%)"
        print(f"\n{'='*70}")
        print(alert_msg)
        print(f"   Marcado el: {marked_date}")
        print(f"   Track ID: {track_id}")
        print(f"{'='*70}\n")
        
        log_alert(alert_msg)
        self.alerts_sent[suspicious_id] = current_time

# =======================
# BUCLE PRINCIPAL
# =======================

# Cargar personas sospechosas
print("🔍 Cargando base de datos de personas sospechosas...")
suspicious_features = load_suspicious_persons_features()

if not suspicious_features:
    print("⚠️  No hay personas marcadas como sospechosas en la base de datos")
    print("   1. Ejecuta primero: python detection_system.py")
    print("   2. Luego ejecuta: python web_interface.py")
    print("   3. Marca personas como sospechosas en la web")
    print("   4. Finalmente ejecuta este script")
    exit()

print(f"✅ Cargadas {len(suspicious_features)} personas sospechosas")
for susp_id, data in suspicious_features.items():
    print(f"   - {data['name']} (Marcado: {data['marked_date']})")

# Abrir video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir '{VIDEO_PATH}'")
    exit()

# Crear tracker
tracker = SuspiciousPersonTracker(suspicious_features)

print("\n" + "="*70)
print("🎥 Sistema de Tracking de Personas Sospechosas Activado")
print("="*70)
print("📋 Funcionalidades:")
print("   ✓ Reconocimiento facial por histograma de color")
print("   ✓ Alertas automáticas cuando detecta sospechoso marcado")
print("   ✓ Registro de alertas en alerts_log.txt")
print("   ✓ Tracking especial con recuadro rojo")
print("\n⌨️ Controles:")
print("   ESC - Salir | ESPACIO - Pausar | R - Reiniciar")
print("="*70 + "\n")

frame_count = 0
paused = False
alert_frame_counter = 0
current_alert_message = ""

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\n🎬 Video finalizado")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            tracker.tracks.clear()
            continue
        
        frame_count += 1
    
    # Detectar personas
    results = person_model(frame, verbose=False)[0]
    
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > 0.5:
            x1, y1, x2, y2 = box.tolist()
            detections.append((x1, y1, x2, y2, float(conf)))
    
    # Actualizar tracker
    tracks = tracker.update(detections, frame)
    
    # Dibujar tracks
    suspicious_detected = 0
    
    for track_id, track_info in tracks.items():
        if track_info['frames_missing'] > 0:
            continue
        
        x1, y1, x2, y2 = track_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        is_suspicious_match = track_info['matched_suspicious_id'] is not None
        
        if is_suspicious_match:
            suspicious_detected += 1
            # ROJO INTENSO para sospechosos
            color = (0, 0, 255)
            thickness = 5
            
            susp_id = track_info['matched_suspicious_id']
            susp_name = suspicious_features[susp_id]['name']
            similarity = track_info['best_match_similarity']
            
            label = f"{susp_name} ({similarity:.0f}%)"
            
            # Rectángulo parpadeante
            if frame_count % 10 < 5:
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 3)
            
            # Alerta en overlay
            if not track_info['alert_shown']:
                current_alert_message = f"PERSONA SOSPECHOSA: {susp_name}"
                alert_frame_counter = 90  # 3 segundos a 30fps
                track_info['alert_shown'] = True
        else:
            # Verde para personas normales
            color = (0, 255, 0)
            thickness = 2
            label = f"ID:{track_id}"
        
        # Dibujar bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label con fondo
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar alerta si está activa
    if alert_frame_counter > 0:
        show_alert_overlay(frame, current_alert_message)
        alert_frame_counter -= 1
    
    # Panel de estadísticas
    cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas en escena: {len([t for t in tracks.values() if t['frames_missing'] == 0])}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Sospechosos detectados: {suspicious_detected}", 
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    status = "PAUSADO" if paused else "ACTIVO"
    cv2.putText(frame, status, (20, 105), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not paused else (255, 165, 0), 1)
    
    cv2.imshow("Tracking de Personas Sospechosas", frame)
    
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPACIO
        paused = not paused
    elif key == ord('r') or key == ord('R'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        tracker.tracks.clear()

cap.release()
cv2.destroyAllWindows()

print("\n✅ Tracking finalizado")
print(f"📄 Revisa el archivo '{ALERTS_LOG}' para ver el historial de alertas")