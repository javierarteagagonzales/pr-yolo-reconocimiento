#!/usr/bin/env python3
"""
Setup Script - Configuración Inicial del Sistema
================================================
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Verifica versión de Python"""
    if sys.version_info < (3, 7):
        print("❌ Se requiere Python 3.7 o superior")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

def check_dependencies():
    """Verifica e instala dependencias"""
    print("\n🔍 Verificando dependencias...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'flask': 'flask',
        'ultralytics': 'ultralytics'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} dependencias")
        print(f"\n💡 Instala con:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("\n✅ Todas las dependencias instaladas")
    return True

def create_directories():
    """Crea directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    dirs = [
        'detected_persons',
        'templates',
        'reports',
        'evidence'
    ]
    
    for dirname in dirs:
        Path(dirname).mkdir(exist_ok=True)
        print(f"✅ {dirname}/")

def initialize_database():
    """Inicializa la base de datos"""
    print("\n🗄️  Inicializando base de datos...")
    
    try:
        from database import initialize_database
        db = initialize_database()
        print("✅ Base de datos inicializada")
        
        # Mostrar estadísticas
        stats = db.get_statistics()
        print(f"\n📊 Estadísticas:")
        print(f"   Personas: {stats['total_persons']}")
        print(f"   Sospechosos: {stats['suspicious_count']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_templates():
    """Crea archivos de templates necesarios"""
    print("\n📄 Verificando templates...")
    
    templates_needed = [
        'dashboard.html',
        'persons.html',
        'reports.html',
        'events.html',
        'settings.html'
    ]
    
    templates_dir = Path('templates')
    
    for template in templates_needed:
        template_path = templates_dir / template
        if template_path.exists():
            print(f"✅ {template}")
        else:
            print(f"⚠️  {template} - se creará al iniciar la app")

def check_video():
    """Verifica si existe el video de prueba"""
    print("\n🎬 Verificando video...")
    
    if Path('test_video.mp4').exists():
        print("✅ test_video.mp4 encontrado")
        return True
    else:
        print("⚠️  test_video.mp4 NO encontrado")
        print("\n💡 Coloca tu video de prueba y renómbralo como:")
        print("   test_video.mp4")
        return False

def print_next_steps():
    """Imprime los siguientes pasos"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETADO")
    print("="*60)
    
    print("\n📝 SIGUIENTES PASOS:")
    print("\n1️⃣  Si no tienes video de prueba:")
    print("   - Coloca un video en la carpeta del proyecto")
    print("   - Renómbralo como: test_video.mp4")
    
    print("\n2️⃣  Iniciar el sistema web:")
    print("   python app.py")
    
    print("\n3️⃣  Abrir navegador en:")
    print("   http://localhost:5000")
    
    print("\n4️⃣  Desde el dashboard web puedes:")
    print("   ✓ Iniciar detección de personas")
    print("   ✓ Marcar personas como sospechosas")
    print("   ✓ Iniciar tracking automático")
    print("   ✓ Generar reportes")
    
    print("\n" + "="*60)

def main():
    """Función principal"""
    print("="*60)
    print("🛡️  SISTEMA DE SEGURIDAD - SETUP")
    print("="*60)
    
    # Verificar Python
    check_python_version()
    
    # Verificar dependencias
    if not check_dependencies():
        print("\n⚠️  Por favor instala las dependencias primero")
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Inicializar base de datos
    if not initialize_database():
        print("\n⚠️  Error inicializando base de datos")
        sys.exit(1)
    
    # Verificar templates
    create_templates()
    
    # Verificar video
    has_video = check_video()
    
    # Imprimir siguientes pasos
    print_next_steps()
    
    if not has_video:
        print("\n⚠️  IMPORTANTE: Necesitas un video para continuar")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelado")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)