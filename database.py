#!/usr/bin/env python3
"""
Sistema de Base de Datos para Personas Detectadas
==================================================
Usa SQLite para persistencia real de datos
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os

DATABASE_FILE = "security_system.db"

class SecurityDatabase:
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self.conn = None
        self.create_tables()
    
    def connect(self):
        """Conecta a la base de datos"""
        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        return self.conn
    
    def close(self):
        """Cierra la conexión"""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Crea todas las tablas necesarias"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Tabla de personas detectadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT UNIQUE NOT NULL,
                track_id INTEGER,
                image_path TEXT,
                timestamp INTEGER,
                detection_date TEXT,
                is_suspicious INTEGER DEFAULT 0,
                frames_tracked INTEGER,
                duration REAL,
                confidence REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de eventos/alertas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                person_id TEXT,
                camera_id TEXT,
                timestamp TEXT,
                description TEXT,
                severity TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_by TEXT,
                resolved_at TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        # Tabla de tracking (historial de movimientos)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                camera_id TEXT,
                position_x REAL,
                position_y REAL,
                timestamp TEXT,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        # Tabla de reportes generados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE,
                report_type TEXT,
                generated_at TEXT,
                generated_by TEXT,
                file_path TEXT,
                metadata TEXT
            )
        """)
        
        # Tabla de configuración del sistema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Base de datos inicializada: {self.db_file}")
    
    # ==========================================
    # OPERACIONES DE PERSONAS
    # ==========================================
    
    def add_person(self, person_data):
        """Agrega o actualiza una persona"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Verificar si ya existe
            cursor.execute("SELECT id, is_suspicious FROM persons WHERE person_id = ?", 
                         (person_data['person_id'],))
            existing = cursor.fetchone()
            
            if existing:
                # ACTUALIZAR manteniendo is_suspicious
                cursor.execute("""
                    UPDATE persons SET
                        track_id = ?,
                        image_path = ?,
                        timestamp = ?,
                        detection_date = ?,
                        frames_tracked = ?,
                        duration = ?,
                        confidence = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE person_id = ?
                """, (
                    person_data.get('track_id'),
                    person_data.get('image_path'),
                    person_data.get('timestamp'),
                    person_data.get('date'),
                    person_data.get('frames_tracked'),
                    person_data.get('duration'),
                    person_data.get('confidence', 0.0),
                    person_data['person_id']
                ))
                
                print(f"✅ Persona actualizada (manteniendo estado): {person_data['person_id']}")
            else:
                # INSERTAR nueva
                cursor.execute("""
                    INSERT INTO persons (
                        person_id, track_id, image_path, timestamp, 
                        detection_date, is_suspicious, frames_tracked, 
                        duration, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    person_data['person_id'],
                    person_data.get('track_id'),
                    person_data.get('image_path'),
                    person_data.get('timestamp'),
                    person_data.get('date'),
                    person_data.get('is_suspicious', 0),
                    person_data.get('frames_tracked'),
                    person_data.get('duration'),
                    person_data.get('confidence', 0.0)
                ))
                
                print(f"✅ Persona insertada: {person_data['person_id']}")
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error guardando persona: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_person(self, person_id):
        """Obtiene una persona por ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_persons(self):
        """Obtiene todas las personas"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM persons ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def mark_as_suspicious(self, person_id, is_suspicious=True, notes=None):
        """Marca una persona como sospechosa"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE persons 
                SET is_suspicious = ?, 
                    notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
            """, (1 if is_suspicious else 0, notes, person_id))
            
            conn.commit()
            
            # Registrar evento
            self.add_event({
                'event_type': 'suspicious_marked' if is_suspicious else 'suspicious_unmarked',
                'person_id': person_id,
                'description': f"Persona marcada como {'sospechosa' if is_suspicious else 'normal'}",
                'severity': 'HIGH' if is_suspicious else 'LOW'
            })
            
            print(f"✅ Persona {person_id} marcada como {'sospechosa' if is_suspicious else 'normal'}")
            return True
            
        except Exception as e:
            print(f"❌ Error marcando persona: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_suspicious_persons(self):
        """Obtiene todas las personas sospechosas"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM persons WHERE is_suspicious = 1 ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ==========================================
    # OPERACIONES DE EVENTOS
    # ==========================================
    
    def add_event(self, event_data):
        """Registra un evento"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO events (
                    event_type, person_id, camera_id, timestamp,
                    description, severity, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_data.get('event_type'),
                event_data.get('person_id'),
                event_data.get('camera_id', 'cam_001'),
                event_data.get('timestamp', datetime.now().isoformat()),
                event_data.get('description'),
                event_data.get('severity', 'MEDIUM'),
                json.dumps(event_data.get('metadata', {}))
            ))
            
            conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            print(f"❌ Error registrando evento: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_recent_events(self, limit=50):
        """Obtiene eventos recientes"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM events 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ==========================================
    # ESTADÍSTICAS
    # ==========================================
    
    def get_statistics(self):
        """Obtiene estadísticas del sistema"""
        conn = self.connect()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total de personas
        cursor.execute("SELECT COUNT(*) as total FROM persons")
        stats['total_persons'] = cursor.fetchone()['total']
        
        # Personas sospechosas
        cursor.execute("SELECT COUNT(*) as total FROM persons WHERE is_suspicious = 1")
        stats['suspicious_count'] = cursor.fetchone()['total']
        
        # Eventos totales
        cursor.execute("SELECT COUNT(*) as total FROM events")
        stats['total_events'] = cursor.fetchone()['total']
        
        # Eventos no resueltos
        cursor.execute("SELECT COUNT(*) as total FROM events WHERE resolved = 0")
        stats['unresolved_events'] = cursor.fetchone()['total']
        
        # Última detección
        cursor.execute("SELECT MAX(detection_date) as last_detection FROM persons")
        stats['last_detection'] = cursor.fetchone()['last_detection']
        
        conn.close()
        
        return stats
    
    # ==========================================
    # MIGRACIÓN DESDE JSON
    # ==========================================
    
    def migrate_from_json(self, json_file="persons_data.json"):
        """Migra datos desde JSON a SQL"""
        if not os.path.exists(json_file):
            print(f"⚠️  Archivo {json_file} no existe")
            return
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            migrated = 0
            for person_id, data in json_data.items():
                data['person_id'] = person_id
                if self.add_person(data):
                    migrated += 1
            
            print(f"✅ Migradas {migrated} personas desde JSON")
            
        except Exception as e:
            print(f"❌ Error migrando datos: {e}")
    
    # ==========================================
    # EXPORTAR A JSON (compatibilidad)
    # ==========================================
    
    def export_to_json(self, json_file="persons_data.json"):
        """Exporta datos a JSON para compatibilidad"""
        persons = self.get_all_persons()
        
        json_data = {}
        for person in persons:
            person_id = person['person_id']
            json_data[person_id] = {
                'id': person_id,
                'track_id': person['track_id'],
                'image_path': person['image_path'],
                'timestamp': person['timestamp'],
                'date': person['detection_date'],
                'is_suspicious': bool(person['is_suspicious']),
                'frames_tracked': person['frames_tracked'],
                'duration': person['duration']
            }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exportadas {len(json_data)} personas a JSON")

# ==========================================
# FUNCIONES DE UTILIDAD
# ==========================================

def initialize_database():
    """Inicializa la base de datos"""
    db = SecurityDatabase()
    
    # Migrar datos existentes si hay JSON
    if os.path.exists("persons_data.json"):
        print("📦 Migrando datos desde persons_data.json...")
        db.migrate_from_json()
    
    return db

def test_database():
    """Prueba la base de datos"""
    print("\n🧪 Probando base de datos...\n")
    
    db = SecurityDatabase()
    
    # Test 1: Agregar persona
    test_person = {
        'person_id': 'test_person_001',
        'track_id': 1,
        'image_path': 'test/image.jpg',
        'timestamp': int(datetime.now().timestamp()),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'frames_tracked': 100,
        'duration': 5.5,
        'is_suspicious': False
    }
    
    db.add_person(test_person)
    
    # Test 2: Marcar como sospechoso
    db.mark_as_suspicious('test_person_001', True, "Comportamiento sospechoso")
    
    # Test 3: Obtener persona
    person = db.get_person('test_person_001')
    print(f"Persona obtenida: {person}")
    
    # Test 4: Estadísticas
    stats = db.get_statistics()
    print(f"\n📊 Estadísticas:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Tests completados")

if __name__ == "__main__":
    # Inicializar base de datos
    initialize_database()
    
    # Ejecutar tests
    # test_database()