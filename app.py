#!/usr/bin/env python3
"""
Dashboard Web Completo - Sistema de Seguridad
==============================================
Interfaz web con menú, botones ejecutables y reportes
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import threading
from database import SecurityDatabase, initialize_database

app = Flask(__name__)

# Configuración
DETECTED_PERSONS_DIR = "detected_persons"
TEMPLATES_DIR = "templates"
REPORTS_DIR = "reports"

# Crear directorios
Path(TEMPLATES_DIR).mkdir(exist_ok=True)
Path(REPORTS_DIR).mkdir(exist_ok=True)

# Inicializar base de datos
db = initialize_database()

# Estado de los procesos en ejecución
running_processes = {
    'detection': None,
    'tracking': None
}

# ==========================================
# RUTAS PRINCIPALES
# ==========================================

@app.route('/')
def index():
    """Dashboard principal"""
    return render_template('dashboard.html')

@app.route('/persons')
def persons_page():
    """Página de gestión de personas"""
    return render_template('persons.html')

@app.route('/reports')
def reports_page():
    """Página de reportes"""
    return render_template('reports.html')

@app.route('/events')
def events_page():
    """Página de eventos"""
    return render_template('events.html')

@app.route('/settings')
def settings_page():
    """Página de configuración"""
    return render_template('settings.html')

# ==========================================
# API - PERSONAS
# ==========================================

@app.route('/api/persons', methods=['GET'])
def get_persons():
    """Obtiene todas las personas"""
    try:
        persons = db.get_all_persons()
        
        # Convertir a formato compatible con frontend
        result = {}
        for person in persons:
            person_id = person['person_id']
            result[person_id] = {
                'id': person_id,
                'track_id': person['track_id'],
                'image_path': person['image_path'],
                'timestamp': person['timestamp'],
                'date': person['detection_date'],
                'is_suspicious': bool(person['is_suspicious']),
                'frames_tracked': person['frames_tracked'],
                'duration': person['duration'],
                'notes': person['notes']
            }
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/persons/suspicious', methods=['GET'])
def get_suspicious():
    """Obtiene personas sospechosas"""
    try:
        persons = db.get_suspicious_persons()
        return jsonify(persons)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/persons/<person_id>/mark', methods=['POST'])
def mark_person(person_id):
    """Marca/desmarca persona como sospechosa"""
    try:
        data = request.json
        is_suspicious = data.get('is_suspicious', True)
        notes = data.get('notes', '')
        
        success = db.mark_as_suspicious(person_id, is_suspicious, notes)
        
        # Exportar a JSON para compatibilidad
        db.export_to_json()
        
        if success:
            return jsonify({
                'success': True,
                'person_id': person_id,
                'is_suspicious': is_suspicious
            })
        else:
            return jsonify({'success': False, 'error': 'Database error'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==========================================
# API - ESTADÍSTICAS
# ==========================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtiene estadísticas del sistema"""
    try:
        stats = db.get_statistics()
        
        # Agregar estadísticas adicionales
        stats['normal_count'] = stats['total_persons'] - stats['suspicious_count']
        
        # Estado de procesos
        stats['detection_running'] = running_processes['detection'] is not None
        stats['tracking_running'] = running_processes['tracking'] is not None
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# API - EVENTOS
# ==========================================

@app.route('/api/events', methods=['GET'])
def get_events():
    """Obtiene eventos recientes"""
    try:
        limit = request.args.get('limit', 50, type=int)
        events = db.get_recent_events(limit)
        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# API - CONTROL DE PROCESOS
# ==========================================

@app.route('/api/process/detection/start', methods=['POST'])
def start_detection():
    """Inicia el proceso de detección"""
    try:
        if running_processes['detection'] is not None:
            return jsonify({
                'success': False,
                'error': 'El proceso de detección ya está en ejecución'
            }), 400
        
        # Iniciar proceso en thread separado
        def run_detection():
            global running_processes
            try:
                process = subprocess.Popen(
                    [sys.executable, 'detection_system.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                running_processes['detection'] = process
                process.wait()
            except Exception as e:
                print(f"❌ Error en detección: {e}")
            finally:
                running_processes['detection'] = None
        
        thread = threading.Thread(target=run_detection, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Proceso de detección iniciado'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process/detection/stop', methods=['POST'])
def stop_detection():
    """Detiene el proceso de detección"""
    try:
        if running_processes['detection'] is None:
            return jsonify({
                'success': False,
                'error': 'No hay proceso de detección en ejecución'
            }), 400
        
        running_processes['detection'].terminate()
        running_processes['detection'] = None
        
        return jsonify({
            'success': True,
            'message': 'Proceso de detección detenido'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process/tracking/start', methods=['POST'])
def start_tracking():
    """Inicia el proceso de tracking"""
    try:
        # Verificar si hay sospechosos
        suspicious = db.get_suspicious_persons()
        if len(suspicious) == 0:
            return jsonify({
                'success': False,
                'error': 'No hay personas marcadas como sospechosas'
            }), 400
        
        if running_processes['tracking'] is not None:
            return jsonify({
                'success': False,
                'error': 'El proceso de tracking ya está en ejecución'
            }), 400
        
        # Iniciar tracking
        def run_tracking():
            global running_processes
            try:
                # En Windows abre nueva ventana
                if sys.platform == 'win32':
                    process = subprocess.Popen(
                        [sys.executable, 'tracking_marked_suspicious.py'],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, 'tracking_marked_suspicious.py']
                    )
                running_processes['tracking'] = process
                process.wait()
            except Exception as e:
                print(f"❌ Error en tracking: {e}")
            finally:
                running_processes['tracking'] = None
        
        thread = threading.Thread(target=run_tracking, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Tracking iniciado para {len(suspicious)} personas',
            'suspicious_count': len(suspicious)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process/tracking/stop', methods=['POST'])
def stop_tracking():
    """Detiene el proceso de tracking"""
    try:
        if running_processes['tracking'] is None:
            return jsonify({
                'success': False,
                'error': 'No hay proceso de tracking en ejecución'
            }), 400
        
        running_processes['tracking'].terminate()
        running_processes['tracking'] = None
        
        return jsonify({
            'success': True,
            'message': 'Proceso de tracking detenido'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==========================================
# API - REPORTES
# ==========================================

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Genera un reporte"""
    try:
        data = request.json
        report_type = data.get('type', 'general')
        
        # Generar reporte
        report_data = {
            'report_id': f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'type': report_type,
            'generated_at': datetime.now().isoformat(),
            'stats': db.get_statistics(),
            'suspicious_persons': db.get_suspicious_persons(),
            'recent_events': db.get_recent_events(20)
        }
        
        # Guardar reporte
        report_file = os.path.join(REPORTS_DIR, f"{report_data['report_id']}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'report_id': report_data['report_id'],
            'file_path': report_file
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reports/list', methods=['GET'])
def list_reports():
    """Lista todos los reportes"""
    try:
        reports = []
        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(REPORTS_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    reports.append({
                        'report_id': report_data['report_id'],
                        'type': report_data['type'],
                        'generated_at': report_data['generated_at'],
                        'file_path': filepath
                    })
        
        return jsonify(reports)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    """Obtiene un reporte específico"""
    try:
        report_file = os.path.join(REPORTS_DIR, f"{report_id}.json")
        
        if not os.path.exists(report_file):
            return jsonify({'error': 'Reporte no encontrado'}), 404
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        return jsonify(report_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# SERVIR ARCHIVOS
# ==========================================

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sirve imágenes"""
    try:
        return send_from_directory(DETECTED_PERSONS_DIR, filename)
    except Exception as e:
        return "Image not found", 404

# ==========================================
# INICIALIZACIÓN
# ==========================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 DASHBOARD WEB - SISTEMA DE SEGURIDAD")
    print("="*60)
    
    # Verificar base de datos
    stats = db.get_statistics()
    print(f"\n📊 Estadísticas:")
    print(f"   Total de personas: {stats['total_persons']}")
    print(f"   Sospechosas: {stats['suspicious_count']}")
    print(f"   Eventos: {stats['total_events']}")
    
    print("\n" + "="*60)
    print("📱 Abre tu navegador en: http://localhost:5000")
    print("="*60)
    print("\n💡 Funcionalidades:")
    print("   ✓ Dashboard con estadísticas en tiempo real")
    print("   ✓ Control de detección y tracking desde web")
    print("   ✓ Gestión de personas y marcado de sospechosos")
    print("   ✓ Generación de reportes")
    print("   ✓ Timeline de eventos")
    print("\n⚠️  Para detener: Ctrl+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)