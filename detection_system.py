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
from database import SecurityDatabase

# Inicializar base de datos
db = SecurityDatabase()

# =======================
# CONFIGURACIÓN
# =======================

# Directorios
DETECTED_PERSONS_DIR = "detected_persons"
DATA_FILE = "persons_data.json"

# Crear directorios si no existen
Path(DETECTED_PERSONS_DIR).mkdir(exist_ok=True)

# Cargar o crear archivo de datos (MANTIENE ESTADO PREVIO)
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        persons_database = json.load(f)
    print(f"✅ Base de datos cargada: {len(persons_database)} personas")
    suspicious_count = sum(1 for p in persons_database.values() if p.get('is_suspicious', False))
    if suspicious_count > 0:
        print(f"   🚨 {suspicious_count} personas marcadas como sospechosas")
else:
    persons_database = {}
    print("📝 Creando nueva base de datos")

# Modelo YOLO
print("🔄 Cargando modelo YOLO...")
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")
print("✅ Modelo cargado")

# Clasificadores Haar Cascade (incluidos con OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parámetros de detección (AJUSTADOS PARA GUARDAR MÁS FÁCILMENTE)
PERSON_CONFIDENCE = 0.4        # Reducido para detectar más personas
MIN_FRAMES_TO_SAVE = 3         # Reducido para guardar más rápido
IMAGE_QUALITY_THRESHOLD = 50   # Reducido para aceptar más imágenes

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
        self.max_distance = 150  # Aumentado para mejor tracking
        
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
                # Guardar antes de eliminar
                if track_info['best_image'] is not None and track_info['database_id'] is None:
                    self.save_track(track_id, track_info)
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
                
                # Intentar guardar si se está perdiendo
                if track_info['frames_missing'] > 15 and track_info['best_image'] is not None:
                    if track_info['database_id'] is None:
                        self.save_track(track_id, track_info)
        
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
                    'database_id': None
                }
                self.next_id += 1
        
        return self.tracks
    
    def save_track(self, track_id, track_info):
        """
        Guarda el track en la base de datos
        """
        if track_info['best_image'] is None:
            return
        
        timestamp = int(time.time())
        person_id = f"person_{timestamp}_{track_id}"
        
        # Guardar imagen
        image_filename = f"{person_id}.jpg"
        image_path = os.path.join(DETECTED_PERSONS_DIR, image_filename)
        
        try:
            cv2.imwrite(image_path, track_info['best_image'])
            print(f"✅ Imagen guardada: {image_path}")
        except Exception as e:
            print(f"❌ Error guardando imagen: {e}")
            return
        
        # Actualizar base de datos (MANTIENE is_suspicious si ya existía)
        existing_suspicious = False
        if person_id in persons_database:
            existing_suspicious = persons_database[person_id].get('is_suspicious', False)
        
        persons_database[person_id] = {
            'id': person_id,
            'track_id': track_id,
            'image_path': image_path,
            'timestamp': timestamp,
            'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            'is_suspicious': existing_suspicious,  # Mantener estado previo
            'frames_tracked': track_info['total_frames'],
            'duration': time.time() - track_info['first_seen']
        }
        
        # Guardar archivo JSON
        try:
            with open(DATA_FILE, 'w') as f:
                json.dump(persons_database, f, indent=2)
            print(f"✅ Persona guardada en BD: {person_id} (Calidad: {track_info['best_quality']:.0f}/100)")
        except Exception as e:
            print(f"❌ Error guardando en BD: {e}")
        
        track_info['database_id'] = person_id

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
    sharpness_score = min(laplacian_var / 5, 40)  # Menos estricto
    quality_score += sharpness_score
    
    # 2. ILUMINACIÓN - Evitar muy oscuro o muy claro
    mean_brightness = np.mean(gray)
    if 60 < mean_brightness < 180:
        quality_score += 30  # Iluminación ideal
    elif 30 < mean_brightness < 220:  # Más permisivo
        quality_score += 20  # Iluminación aceptable
    
    # 3. TAMAÑO - Preferir imágenes más grandes
    size_score = min(image.shape[0] * image.shape[1] / 3000, 30)  # Menos estricto
    quality_score += size_score
    
    return min(quality_score, 100)

def has_clear_face(image):
    """
    Verifica si la imagen tiene un rostro claro visible
    """
    if image.size == 0:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))  # Menos estricto
    return len(faces) > 0

# =======================
# PROCESAMIENTO DE PERSONAS
# =======================

def process_person_detection(track_id, person_roi, track_info):
    """
    Procesa detección de persona y actualiza mejor imagen
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
        quality += 15
    
    # DEBUG: Imprimir calidad
    if track_info['total_frames'] % 10 == 0:  # Cada 10 frames
        print(f"🔍 Track {track_id}: Calidad actual={quality:.0f}, Mejor={track_info['best_quality']:.0f}")
    
    # Guardar si es mejor que la anterior
    if quality > track_info['best_quality']:
        track_info['best_image'] = person_roi.copy()
        track_info['best_quality'] = quality
        print(f"📸 Track {track_id}: Nueva mejor imagen (Q={quality:.0f})")

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
print("\nℹ️  Configuración ajustada para guardar más imágenes:")
print(f"   • Confianza mínima: {PERSON_CONFIDENCE}")
print(f"   • Frames mínimos: {MIN_FRAMES_TO_SAVE}")
print(f"   • Calidad mínima: {IMAGE_QUALITY_THRESHOLD}/100")
print("\n⌨️ Controles:")
print("   ESC - Salir y guardar todas las personas")
print("   ESPACIO - Pausar/Reanudar")
print("   S - Forzar guardado de todas las personas ahora")
print("="*60)
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
            print("\n🎬 Video finalizado - Guardando personas...")
            # Guardar todos los tracks pendientes
            for track_id, track_info in tracker.tracks.items():
                if track_info['best_image'] is not None and track_info['database_id'] is None:
                    if track_info['best_quality'] >= IMAGE_QUALITY_THRESHOLD:
                        tracker.save_track(track_id, track_info)
            break
        
        frame_count += 1
    
    # Detectar personas con YOLO
    results = person_model(frame, verbose=False)[0]
    
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > PERSON_CONFIDENCE:  # Clase 0 = persona
            x1, y1, x2, y2 = box.tolist()
            
            # Validar dimensiones mínimas (menos estricto)
            width, height = x2 - x1, y2 - y1
            if height < 80 or width < 30:
                continue
            
            detections.append((x1, y1, x2, y2, float(conf)))
    
    # Actualizar tracker
    tracks = tracker.update(detections)
    
    tracked_count = 0
    saved_count = 0
    
    # Procesar cada track
    for track_id, track_info in tracks.items():
        if track_info['frames_missing'] > 0:
            continue
        
        tracked_count += 1
        x1, y1, x2, y2 = track_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extraer ROI de la persona
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        
        # Procesar y actualizar mejor imagen
        process_person_detection(track_id, person_roi, track_info)
        
        # Contar guardadas
        if track_info['database_id']:
            saved_count += 1
        
        # Verificar si debe guardar ahora (cada 50 frames)
        if track_info['total_frames'] % 50 == 0 and track_info['database_id'] is None:
            if track_info['best_image'] is not None and track_info['best_quality'] >= IMAGE_QUALITY_THRESHOLD:
                tracker.save_track(track_id, track_info)
        
        # Determinar color
        if track_info['database_id']:
            color = (0, 165, 255)  # NARANJA - Ya guardada
            label = f"ID:{track_id} GUARDADO"
            thickness = 3
        else:
            color = (0, 255, 0)  # VERDE - Trackeando
            label = f"ID:{track_id}"
            thickness = 2
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Mostrar calidad de imagen actual
        quality_text = f"Q: {track_info['best_quality']:.0f}/100 F:{track_info['total_frames']}"
        cv2.putText(frame, quality_text, (x1, y2+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Panel de estadísticas en pantalla
    cv2.rectangle(frame, (10, 10), (500, 170), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Personas en escena: {tracked_count}", 
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Guardadas (naranja): {saved_count}", 
               (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(frame, f"Total en BD: {len(persons_database)}", 
               (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status = "PAUSADO" if paused else "Reproduciendo"
    cv2.putText(frame, status, (20, 155), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Mostrar frame
    cv2.imshow("Sistema de Deteccion y Registro", frame)
    
    # Controles de teclado
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    
    if key == 27:  # ESC - Salir y guardar todo
        print("\n💾 Guardando todas las personas detectadas...")
        for track_id, track_info in tracker.tracks.items():
            if track_info['best_image'] is not None and track_info['database_id'] is None:
                if track_info['best_quality'] >= IMAGE_QUALITY_THRESHOLD:
                    tracker.save_track(track_id, track_info)
        break
    elif key == 32:  # ESPACIO - Pausar/Reanudar
        paused = not paused
    elif key == ord('s') or key == ord('S'):  # S - Guardar ahora
        print("\n💾 Forzando guardado de todas las personas...")
        for track_id, track_info in tracker.tracks.items():
            if track_info['best_image'] is not None and track_info['database_id'] is None:
                tracker.save_track(track_id, track_info)

cap.release()
cv2.destroyAllWindows()

# =======================
# RESUMEN FINAL
# =======================

print("\n" + "="*60)
print("✅ Detección finalizada")
print("="*60)
print(f"📊 Total de personas guardadas: {len(persons_database)}")
print(f"📁 Imágenes guardadas en: {DETECTED_PERSONS_DIR}/")
print(f"📄 Base de datos: {DATA_FILE}")

# Verificar si hay imágenes
if len(persons_database) == 0:
    print("\n⚠️  NO SE GUARDÓ NINGUNA PERSONA")
    print("\n🔍 Posibles causas:")
    print("   1. El video no tiene personas visibles")
    print("   2. Las personas están muy lejos o borrosas")
    print("   3. El video es muy corto")
    print("\n💡 Soluciones:")
    print("   • Reduce IMAGE_QUALITY_THRESHOLD a 30 (línea 44)")
    print("   • Reduce MIN_FRAMES_TO_SAVE a 1 (línea 43)")
    print("   • Usa un video con personas más cerca de la cámara")
else:
    print(f"\n✅ Imágenes guardadas correctamente")
    print(f"   {len(os.listdir(DETECTED_PERSONS_DIR))} archivos en {DETECTED_PERSONS_DIR}/")
    print("\n💡 Siguiente paso:")
    print("   Ejecuta: python web_interface.py")
    print("   Para abrir la interfaz web")
print("="*60)