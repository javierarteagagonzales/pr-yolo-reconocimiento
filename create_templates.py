#!/usr/bin/env python3
"""
Script para crear todos los templates HTML necesarios
"""

import os
from pathlib import Path

# Crear directorio templates
Path('templates').mkdir(exist_ok=True)

print("📁 Creando templates HTML...")
print("\n⚠️  IMPORTANTE:")
print("Este script solo crea los archivos vacíos.")
print("Debes copiar el contenido de los artifacts a cada archivo:\n")

templates = [
    'dashboard.html',
    'persons.html',
    'events.html',
    'reports.html',
    'settings.html'
]

for template in templates:
    filepath = Path('templates') / template
    
    if filepath.exists():
        print(f"✅ {template} - Ya existe")
    else:
        # Crear archivo vacío con comentario
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"""<!-- 
{template}
=========================================
Copia aquí el contenido del artifact:
- dashboard.html
- persons.html
- events.html
- reports.html
- settings.html
=========================================
-->
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{template.replace('.html', '').title()}</title>
</head>
<body>
    <h1>Template: {template}</h1>
    <p>Copia el contenido del artifact aquí</p>
</body>
</html>
""")
        print(f"📝 {template} - Creado (vacío)")

print("\n" + "="*60)
print("📝 PRÓXIMOS PASOS:")
print("="*60)
print("\n1. Copia el contenido de cada artifact a su archivo:")
print("   • dashboard_html    → templates/dashboard.html")
print("   • persons_html      → templates/persons.html")
print("   • events_html       → templates/events.html")
print("   • reports_html      → templates/reports.html")
print("   • settings_html     → templates/settings.html")
print("\n2. O ejecuta:")
print("   python app.py")
print("   (Algunos templates se crean automáticamente)")
print("\n" + "="*60)