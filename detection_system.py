# detection_system.py
"""
PASO 1: Sistema de Detección y Guardado de Personas
====================================================
Este script:
- Lee un video y detecta todas las personas
- Guarda la mejor imagen de cada persona (más nítida/clara)
- Crea una base de datos en persons_data.json
- Las imágenes se guardan en detected_persons/
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict
import time
import json
import os
from pathlib import Path

# =======================
# CONFIGURACIÓN
# =======================

# Directorios
DETECTED_PERSONS_DIR = "detected_persons"
DATA_FILE = "persons_data.json"

# Crear directorios si no existen
Path(DETECTED_PERSONS_DIR).mkdir(exist_ok=True)

# Cargar o crear archivo de datos
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        persons_database = json.load(f)
else:
    persons_database = {}

# Modelo YOLO
print("🔄 Cargando modelo YOLO...")
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")
print("✅ Modelo cargado")

# Clasificadores Haar Cascade (incluidos con OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parámetros de detección
PERSON_CONFIDENCE = 0.5        # Confianza mínima para detectar persona
MIN_FRAMES_TO_SAVE = 5         # Frames mínimos de tracking antes de guardar
IMAGE_QUALITY_THRESHOLD = 80   # Umbral de calidad (0-100)

# =======================
# SISTEMA DE TRACKING
# =======================

class PersonTracker:
    """
    Sistema simple de tracking de personas usando distancia euclidiana
    """
    def __init__(self):
        self.tracks = {}
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
        
        matched_detections = set()
        
        # Actualizar tracks existentes
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
                matched_detections.add(best_match)
            else:
                self.tracks[track_id]['frames_missing'] += 1
        
        # Crear nuevos tracks
        for idx, (cx, cy, det) in enumerate(current_centers):
            if idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'center': (cx, cy),
                    'bbox': det[:4],
                    'conf': det[4],
                    'frames_missing': 0,
                    'total_frames': 1,
                    'first_seen': time.time(),
                    'best_image': None,
                    'best_quality': 0,
                    'images_saved': 0,
                    'is_marked_suspicious': False,
                    'database_id': None
                }
                self.next_id += 1
        
        return self.tracks

tracker = PersonTracker()

# =======================
# EVALUACIÓN DE CALIDAD DE IMAGEN
# =======================

def calculate_image_quality(image):
    """
    Calcula la calidad de una imagen (0-100)
    Criterios: nitidez, iluminación, tamaño
    """
    if image.size == 0:
        return 0
    
    quality_score = 0
    
    # 1. NITIDEZ - Usando varianza de Laplaciano
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 10, 40)
    quality_score += sharpness_score
    
    # 2. ILUMINACIÓN - Evitar muy oscuro o muy claro
    mean_brightness = np.mean(gray)
    if 60 < mean_brightness < 180:
        quality_score += 30  # Iluminación ideal
    elif 40 < mean_brightness < 200:
        quality_score += 15  # Iluminación aceptable
    
    # 3. TAMAÑO - Preferir imágenes más grandes
    size_score = min(image.shape[0] * image.shape[1] / 5000, 30)
    quality_score += size_score
    
    return min(quality_score, 100)

def has_clear_face(image):
    """
    Verifica si la imagen tiene un rostro claro visible
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    return len(faces) > 0

# =======================
# GESTIÓN DE BASE DE DATOS
# =======================

def save_person_to_database(track_id, image, track_info):
    """
    Guarda la mejor imagen de una persona en la base de datos
    """
    timestamp = int(time.time())
    person_id = f"person_{timestamp}_{track_id}"
    
    # Guardar imagen
    image_filename = f"{person_id}.jpg"
    image_path = os.path.join(DETECTED_PERSONS_DIR, image_filename)
    cv2.imwrite(image_path, image)
    
    # Actualizar base de datos
    persons_database[person_id] = {
        'id': person_id,
        'track_id': track_id,
        'image_path': image_path,
        'timestamp': timestamp,
        'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
        'is_suspicious': False,
        'frames_tracked': track_info['total_frames'],
        'duration': time.time() - track_info['first_seen']
    }
    
    # Guardar archivo JSON
    with open(DATA_FILE, 'w') as f:
        json.dump(persons_database, f, indent=2)
    
    return person_id

def load_suspicious_persons():
    """
    Carga lista de personas marcadas como sospechosas
    """
    return [pid for pid, data in persons_database.items() if data.get('is_suspicious', False)]

# =======================
# PROCESAMIENTO DE PERSONAS
# =======================

def process_person_detection(track_id, person_roi, track_info, frame_count):
    """
    Procesa detección de persona y guarda mejor imagen
    """
    # Solo procesar si ha sido trackeado por suficientes frames
    if track_info['total_frames'] < MIN_FRAMES_TO_SAVE:
        return
    
    if person_roi is None or person_roi.size == 0:
        return
    
    # Calcular calidad de imagen actual
    quality = calculate_image_quality(person_roi)
    
    # Si tiene rostro claro, agregar bonus de calidad
    if has_clear_face(person_roi):
        quality += 20
    
    # Guardar si es mejor que la anterior
    if quality > track_info['best_quality'] and quality >= IMAGE_QUALITY_THRESHOLD:
        track_info['best_image'] = person_roi.copy()
        track_info['best_quality'] = quality
    
    # Guardar en base de datos cuando se pierde el track (sale de escena)
    if track_info['frames_missing'] > 20 and track_info['best_image'] is not None:
        if track_info['database_id'] is None:
            person_id = save_person_to_database(track_id, track_info['best_image'], track_info)
            track_info['database_id'] = person_id
            track_info['images_saved'] += 1
            print(f"✅ Persona guardada: {person_id} (Calidad: {track_info['best_quality']:.0f}/100)")

# =======================
# CONFIGURACIÓN DE VIDEO
# =======================

# OPCIÓN 1: Usar cámara web (comentado por defecto)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# OPCIÓN 2: Usar video local
VIDEO_PATH = "test_video.mp4"  # Cambia esto por tu video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir el video '{VIDEO_PATH}'")
    print("\n📝 Solución:")
    print("   1. Coloca un video en la misma carpeta que este script")
    print("   2. Renombra el video como 'test_video.mp4' o")
    print("   3. Cambia la variable VIDEO_PATH con la ruta correcta")
    print("\n💡 Ejemplos de rutas:")
    print("   Windows:  VIDEO_PATH = 'C:/Videos/mi_video.mp4'")
    print("   Linux:    VIDEO_PATH = '/home/user/videos/mi_video.mp4'")
    print("   Relativa: VIDEO_PATH = 'videos/personas.mp4'")
    exit()

# Obtener información del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print("\n" + "="*60)
print("🎥 Sistema de Detección y Registro de Personas")
print("="*60)
print(f"📹 Video: {VIDEO_PATH}")
print(f"⏱️  Duración: {duration:.1f} segundos ({total_frames} frames a {fps} FPS)")
print(f"📁 Directorio de imágenes: {DETECTED_PERSONS_DIR}/")
print(f"📄 Base de datos: {DATA_FILE}")
print("="*60)
print("\nℹ️  El sistema guardará automáticamente:")
print("   • La mejor imagen de cada persona detectada")
print("   • Solo imágenes de calidad alta (nitidez + iluminación)")
print("   • Preferencia por imágenes con rostro visible")
print("\n⌨️ Controles:")
print("   ESC - Salir y generar interfaz web")
print("   ESPACIO - Pausar/Reanudar")
print("   R - Reiniciar video")
print("="*60)
print()

frame_count = 0
paused = False

# Cargar personas sospechosas marcadas (si existen)
suspicious_list = load_suspicious_persons()

# =======================
# BUCLE PRINCIPAL
# =======================

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\n🎬 Video finalizado")
            break
        
        frame_count += 1
    
    # Detectar personas con YOLO
    results = person_model(frame, verbose=False)[0]
    
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > PERSON_CONFIDENCE:  # Clase 0 = persona
            x1, y1, x2, y2 = box.tolist()
            
            # Validar dimensiones mínimas
            width, height = x2 - x1, y2 - y1
            if height < 100 or width < 40:
                continue
            
            detections.append((x1, y1, x2, y2, float(conf)))
    
    # Actualizar tracker
    tracks = tracker.update(detections)
    
    tracked_count = 0
    suspicious_tracked = 0
    
    # Procesar cada track
    for track_id, track_info in tracks.items():
        if track_info['frames_missing'] > 0:
            # Procesar antes de eliminar (para guardar imagen)
            process_person_detection(track_id, None, track_info, frame_count)
            continue
        
        tracked_count += 1
        x1, y1, x2, y2 = track_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extraer ROI de la persona
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        
        # Procesar y guardar mejor imagen
        process_person_detection(track_id, person_roi, track_info, frame_count)
        
        # Verificar si está marcada como sospechosa
        is_suspicious_marked = (track_info['database_id'] in suspicious_list)
        track_info['is_marked_suspicious'] = is_suspicious_marked
        
        if is_suspicious_marked:
            suspicious_tracked += 1
            color = (0, 0, 255)  # ROJO
            thickness = 4
            label = f"ID:{track_id} SOSPECHOSO MARCADO"
        else:
            color = (0, 255, 0)  # VERDE
            thickness = 2
            label = f"ID:{track_id}"
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Mostrar calidad de imagen actual
        quality_text = f"Q: {track_info['best_quality']:.0f}/100"
        cv2.putText(frame, quality_text, (x1, y2+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Indicador de que ya fue guardada (círculo verde)
        if track_info['database_id']:
            cv2.circle(frame, (x2-10, y1+10), 5, (0, 255, 0), -1)
    
    # Panel de estadísticas en pantalla
    cv2.rectangle(frame, (10, 10), (500, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas en escena: {tracked_count}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Sospechosos marcados: {suspicious_tracked}", 
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total guardados: {len(persons_database)}", 
               (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status = "PAUSADO" if paused else "Reproduciendo"
    cv2.putText(frame, status, (20, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Mostrar frame
    cv2.imshow("Sistema de Deteccion y Registro", frame)
    
    # Controles de teclado
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    
    if key == 27:  # ESC - Salir
        break
    elif key == 32:  # ESPACIO - Pausar/Reanudar
        paused = not paused
    elif key == ord('r') or key == ord('R'):  # R - Reiniciar
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0

cap.release()
cv2.destroyAllWindows()

# =======================
# RESUMEN FINAL
# =======================

print("\n" + "="*60)
print("✅ Detección finalizada")
print("="*60)
print(f"📊 Total de personas guardadas: {len(persons_database)}")
print(f"🚨 Personas marcadas como sospechosas: {len(suspicious_list)}")
print(f"📁 Imágenes guardadas en: {DETECTED_PERSONS_DIR}/")
print(f"📄 Base de datos: {DATA_FILE}")
print("\n💡 Siguiente paso:")
print("   Ejecuta: python web_interface.py")
print("   Para abrir la interfaz web y marcar personas como sospechosas")
print("="*60)