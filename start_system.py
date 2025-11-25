#!/usr/bin/env python3
"""
Script de inicio para el sistema completo
Gestiona la ejecución de detección, interfaz web y tracking
"""

import os
import sys
import json
import subprocess
import time

DATA_FILE = "persons_data.json"

def print_header():
    print("\n" + "="*60)
    print("🎥 SISTEMA DE DETECCIÓN DE PERSONAS SOSPECHOSAS")
    print("="*60)

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    print("\n🔍 Verificando dependencias...")
    
    try:
        import cv2
        import numpy
        from ultralytics import YOLO
        from flask import Flask
        print("✅ Todas las dependencias instaladas")
        return True
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("\n💡 Instala con: pip install ultralytics opencv-python numpy flask")
        return False

def check_video():
    """Verifica que existe el video"""
    print("\n🎬 Verificando video...")
    
    if os.path.exists("test_video.mp4"):
        print("✅ Video encontrado: test_video.mp4")
        return True
    else:
        print("❌ No se encuentra test_video.mp4")
        print("\n💡 Coloca tu video en la carpeta y renómbralo como test_video.mp4")
        return False

def check_database():
    """Verifica si hay personas detectadas"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        total = len(data)
        suspicious = sum(1 for p in data.values() if p.get('is_suspicious', False))
        
        print(f"\n📊 Base de datos encontrada:")
        print(f"   • Total de personas: {total}")
        print(f"   • Sospechosas: {suspicious}")
        
        return total, suspicious
    else:
        print("\n📊 No hay base de datos")
        return 0, 0

def menu_principal():
    """Menú principal del sistema"""
    print_header()
    
    if not check_dependencies():
        return
    
    total, suspicious = check_database()
    
    print("\n" + "="*60)
    print("OPCIONES:")
    print("="*60)
    print("1. 🎥 Detectar personas en video (detection_system.py)")
    print("2. 🌐 Abrir interfaz web para marcar sospechosos")
    print("3. 🎯 Iniciar tracking de sospechosos")
    print("4. 🚀 Modo completo (Web + Auto-tracking)")
    print("5. 📊 Ver estadísticas")
    print("0. ❌ Salir")
    print("="*60)
    
    opcion = input("\nSelecciona una opción: ").strip()
    
    if opcion == "1":
        ejecutar_deteccion()
    elif opcion == "2":
        ejecutar_web()
    elif opcion == "3":
        ejecutar_tracking()
    elif opcion == "4":
        ejecutar_modo_completo()
    elif opcion == "5":
        mostrar_estadisticas()
    elif opcion == "0":
        print("\n👋 ¡Hasta luego!")
        sys.exit(0)
    else:
        print("\n❌ Opción inválida")
        time.sleep(2)
        menu_principal()

def ejecutar_deteccion():
    """Ejecuta el sistema de detección"""
    print("\n🎥 Iniciando detección de personas...")
    
    if not check_video():
        input("\nPresiona Enter para volver al menú...")
        menu_principal()
        return
    
    print("\n💡 Se abrirá una ventana con el video")
    print("   Controles: ESC=Salir | ESPACIO=Pausar | S=Guardar ahora")
    print("\n⏳ Ejecutando detection_system.py...\n")
    
    try:
        subprocess.run([sys.executable, "detection_system.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Detección interrumpida")
    
    input("\nPresiona Enter para volver al menú...")
    menu_principal()

def ejecutar_web():
    """Ejecuta la interfaz web"""
    print("\n🌐 Iniciando interfaz web...")
    
    total, _ = check_database()
    
    if total == 0:
        print("\n⚠️  No hay personas detectadas")
        print("   Ejecuta primero la opción 1 (Detectar personas)")
        input("\nPresiona Enter para volver al menú...")
        menu_principal()
        return
    
    print("\n💡 La interfaz web se abrirá en: http://localhost:5000")
    print("   Presiona Ctrl+C para detener el servidor")
    print("\n⏳ Ejecutando web_interface.py...\n")
    
    try:
        subprocess.run([sys.executable, "web_interface.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Servidor web detenido")
    
    input("\nPresiona Enter para volver al menú...")
    menu_principal()

def ejecutar_tracking():
    """Ejecuta el tracking de sospechosos"""
    print("\n🎯 Iniciando tracking de sospechosos...")
    
    total, suspicious = check_database()
    
    if suspicious == 0:
        print("\n⚠️  No hay personas marcadas como sospechosas")
        print("   Ejecuta la opción 2 para marcar personas en la web")
        input("\nPresiona Enter para volver al menú...")
        menu_principal()
        return
    
    print(f"\n✅ Se hará tracking de {suspicious} personas sospechosas")
    print("\n💡 Se abrirá una ventana con el video y alertas")
    print("   Controles: ESC=Salir | ESPACIO=Pausar | R=Reiniciar")
    print("\n⏳ Ejecutando tracking_marked_suspicious.py...\n")
    
    try:
        subprocess.run([sys.executable, "tracking_marked_suspicious.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Tracking interrumpido")
    
    input("\nPresiona Enter para volver al menú...")
    menu_principal()

def ejecutar_modo_completo():
    """Modo completo: Web + tracking automático"""
    print("\n🚀 Modo completo activado")
    print("\n💡 Se abrirá:")
    print("   1. Interfaz web para marcar sospechosos")
    print("   2. Al marcar alguien, se iniciará tracking automático")
    
    total, _ = check_database()
    
    if total == 0:
        print("\n⚠️  No hay personas detectadas")
        print("   Ejecuta primero la opción 1 (Detectar personas)")
        input("\nPresiona Enter para volver al menú...")
        menu_principal()
        return
    
    print("\n⏳ Iniciando interfaz web...")
    print("   Abre: http://localhost:5000")
    print("   Marca personas y haz clic en 'Iniciar Tracking'")
    
    try:
        subprocess.run([sys.executable, "web_interface.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Sistema detenido")
    
    input("\nPresiona Enter para volver al menú...")
    menu_principal()

def mostrar_estadisticas():
    """Muestra estadísticas del sistema"""
    print("\n📊 ESTADÍSTICAS DEL SISTEMA")
    print("="*60)
    
    total, suspicious = check_database()
    
    if total == 0:
        print("\n⚠️  No hay datos. Ejecuta primero la detección.")
    else:
        print(f"\n✅ Total de personas detectadas: {total}")
        print(f"🚨 Personas sospechosas: {suspicious}")
        print(f"✓  Personas normales: {total - suspicious}")
        
        if os.path.exists("detected_persons"):
            images = len(os.listdir("detected_persons"))
            print(f"📸 Imágenes guardadas: {images}")
        
        if os.path.exists("alerts_log.txt"):
            with open("alerts_log.txt", 'r') as f:
                alerts = len(f.readlines())
            print(f"📝 Alertas registradas: {alerts}")
    
    print("="*60)
    input("\nPresiona Enter para volver al menú...")
    menu_principal()

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n👋 Sistema finalizado")
        sys.exit(0)