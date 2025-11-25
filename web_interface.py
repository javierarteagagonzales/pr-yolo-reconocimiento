# web_interface.py
"""
PASO 2: Interfaz Web para Gestión de Personas Detectadas
==========================================================
Servidor Flask que muestra galería de personas y permite marcarlas como sospechosas
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
from pathlib import Path

app = Flask(__name__)

DATA_FILE = "persons_data.json"
DETECTED_PERSONS_DIR = "detected_persons"
TEMPLATES_DIR = "templates"

# Crear directorio de templates
Path(TEMPLATES_DIR).mkdir(exist_ok=True)

# =======================
# RUTAS API
# =======================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/persons')
def get_persons():
    """Obtiene lista de todas las personas detectadas"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                persons = json.load(f)
            
            # DEBUG: Imprimir en consola
            print(f"\n✅ API /api/persons: Devolviendo {len(persons)} personas")
            
            return jsonify(persons)
        else:
            print(f"\n⚠️  API /api/persons: Archivo {DATA_FILE} no existe")
            return jsonify({})
    except Exception as e:
        print(f"\n❌ Error en /api/persons: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mark_suspicious/<person_id>', methods=['POST'])
def mark_suspicious(person_id):
    """Marca una persona como sospechosa"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                persons = json.load(f)
            
            if person_id in persons:
                data = request.json
                is_suspicious = data.get('is_suspicious', True)
                persons[person_id]['is_suspicious'] = is_suspicious
                
                with open(DATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(persons, f, indent=2, ensure_ascii=False)
                
                status = "sospechosa" if is_suspicious else "normal"
                print(f"✅ Persona {person_id} marcada como {status}")
                
                # Si se marca como sospechoso, indicar que hay que iniciar tracking
                if is_suspicious:
                    print(f"🚨 NUEVO SOSPECHOSO: {person_id}")
                    print(f"   Ejecuta: python tracking_marked_suspicious.py")
                
                return jsonify({
                    'success': True, 
                    'person_id': person_id,
                    'is_suspicious': is_suspicious,
                    'should_start_tracking': is_suspicious
                })
            else:
                print(f"⚠️  Persona {person_id} no encontrada en BD")
                return jsonify({'success': False, 'error': 'Person not found'}), 404
        
        return jsonify({'success': False, 'error': 'Database not found'}), 404
    except Exception as e:
        print(f"❌ Error marcando persona: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Obtiene estadísticas generales"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                persons = json.load(f)
            
            total = len(persons)
            suspicious = sum(1 for p in persons.values() if p.get('is_suspicious', False))
            
            return jsonify({
                'total_persons': total,
                'suspicious_count': suspicious,
                'normal_count': total - suspicious
            })
        
        return jsonify({'total_persons': 0, 'suspicious_count': 0, 'normal_count': 0})
    except Exception as e:
        print(f"❌ Error en /api/stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sirve las imágenes de personas detectadas"""
    try:
        print(f"📸 Sirviendo imagen: {filename}")
        return send_from_directory(DETECTED_PERSONS_DIR, filename)
    except Exception as e:
        print(f"❌ Error sirviendo imagen {filename}: {e}")
        return "Image not found", 404

@app.route('/api/start_tracking', methods=['POST'])
def start_tracking():
    """Inicia el script de tracking en segundo plano"""
    try:
        import subprocess
        import sys
        
        # Verificar si hay personas sospechosas
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                persons = json.load(f)
            
            suspicious_count = sum(1 for p in persons.values() if p.get('is_suspicious', False))
            
            if suspicious_count == 0:
                return jsonify({
                    'success': False, 
                    'error': 'No hay personas marcadas como sospechosas'
                }), 400
            
            # Iniciar tracking en nuevo proceso
            if sys.platform == 'win32':
                # Windows
                subprocess.Popen(['python', 'tracking_marked_suspicious.py'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Linux/Mac - abrir nueva terminal
                terminal_commands = [
                    ['gnome-terminal', '--', 'python3', 'tracking_marked_suspicious.py'],  # Ubuntu
                    ['xterm', '-e', 'python3', 'tracking_marked_suspicious.py'],  # Linux genérico
                    ['osascript', '-e', 'tell app "Terminal" to do script "python3 tracking_marked_suspicious.py"']  # Mac
                ]
                
                for cmd in terminal_commands:
                    try:
                        subprocess.Popen(cmd)
                        break
                    except FileNotFoundError:
                        continue
            
            print(f"🚀 Tracking iniciado para {suspicious_count} personas sospechosas")
            
            return jsonify({
                'success': True, 
                'message': f'Tracking iniciado para {suspicious_count} personas',
                'suspicious_count': suspicious_count
            })
        
        return jsonify({'success': False, 'error': 'Database not found'}), 404
    except Exception as e:
        print(f"❌ Error iniciando tracking: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug')
def debug_info():
    """Endpoint de debug para verificar el sistema"""
    try:
        info = {
            'data_file_exists': os.path.exists(DATA_FILE),
            'images_dir_exists': os.path.exists(DETECTED_PERSONS_DIR),
            'persons_count': 0,
            'images_count': 0,
            'persons_list': []
        }
        
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                persons = json.load(f)
                info['persons_count'] = len(persons)
                info['persons_list'] = list(persons.keys())
        
        if os.path.exists(DETECTED_PERSONS_DIR):
            info['images_count'] = len(os.listdir(DETECTED_PERSONS_DIR))
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =======================
# GENERAR HTML MEJORADO
# =======================

html_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Gestión de Personas Detectadas</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }
        
        h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .debug-info {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #666;
        }
        
        .debug-info code {
            background: #fff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            min-width: 180px;
        }
        
        .stat-card h3 {
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        
        .stat-card p {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .filters {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .filter-btn.active {
            background: #667eea;
            color: white;
            transform: scale(1.05);
        }
        
        .filter-btn:not(.active) {
            background: #e0e0e0;
            color: #666;
        }
        
        .filter-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .persons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        
        .person-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: all 0.3s;
            border: 3px solid transparent;
        }
        
        .person-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .person-card.suspicious {
            border-color: #e74c3c;
        }
        
        .person-card .image-container {
            width: 100%;
            height: 300px;
            overflow: hidden;
            background: #f0f0f0;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .person-card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .person-card .image-error {
            color: #999;
            text-align: center;
            padding: 20px;
        }
        
        .suspicious-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: #e74c3c;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .person-info {
            padding: 20px;
        }
        
        .person-info h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .person-info p {
            color: #666;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        
        .person-info p strong {
            color: #333;
        }
        
        .checkbox-container {
            margin-top: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .checkbox-container input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .checkbox-container label {
            cursor: pointer;
            font-weight: 600;
            color: #e74c3c;
            font-size: 1em;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .empty-state h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .person-card {
            animation: fadeIn 0.5s ease-out;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
            color: #667eea;
        }
        
        .error-message {
            background: #ffe6e6;
            border: 2px solid #e74c3c;
            color: #c0392b;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 10px;
        }
        
        .refresh-btn:hover {
            background: #5568d3;
        }
        
        .tracking-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
            transition: all 0.3s;
        }
        
        .tracking-btn:hover {
            background: #c0392b;
            transform: scale(1.05);
        }
        
        .tracking-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: scale(1);
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 300px;
            animation: slideIn 0.3s ease-out;
        }
        
        .notification.success {
            border-left: 5px solid #2ecc71;
        }
        
        .notification.error {
            border-left: 5px solid #e74c3c;
        }
        
        .notification.warning {
            border-left: 5px solid #f39c12;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .tracking-section {
            background: #fff3e0;
            border: 2px solid #ff9800;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        .tracking-section h3 {
            color: #e65100;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 Sistema de Gestión de Personas Detectadas</h1>
            <p style="color: #666; margin-top: 10px;">Revisa y marca personas sospechosas</p>
        </header>
        
        <div class="debug-info" id="debug-info">
            <strong>🔍 Información del Sistema:</strong><br>
            Cargando información...
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3 id="total-persons">0</h3>
                <p>Total Detectadas</p>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <h3 id="suspicious-count">0</h3>
                <p>Sospechosas</p>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
                <h3 id="normal-count">0</h3>
                <p>Normales</p>
            </div>
        </div>
        
        <div class="filters">
            <button class="filter-btn active" onclick="filterPersons('all')">Todas</button>
            <button class="filter-btn" onclick="filterPersons('suspicious')">Sospechosas</button>
            <button class="filter-btn" onclick="filterPersons('normal')">Normales</button>
            <button class="refresh-btn" onclick="forceReload()">🔄 Recargar</button>
        </div>
        
        <div id="persons-container" class="loading">
            Cargando personas detectadas...
        </div>
    </div>

    <script>
        let allPersons = {};
        let currentFilter = 'all';
        
        // Cargar información de debug
        async function loadDebugInfo() {
            try {
                const response = await fetch('/api/debug');
                const info = await response.json();
                
                const debugDiv = document.getElementById('debug-info');
                debugDiv.innerHTML = `
                    <strong>🔍 Información del Sistema:</strong><br>
                    • Archivo de datos: <code>${info.data_file_exists ? '✅ Existe' : '❌ No existe'}</code><br>
                    • Directorio de imágenes: <code>${info.images_dir_exists ? '✅ Existe' : '❌ No existe'}</code><br>
                    • Personas en BD: <code>${info.persons_count}</code><br>
                    • Imágenes disponibles: <code>${info.images_count}</code>
                `;
                
                console.log('📊 Debug Info:', info);
            } catch (error) {
                console.error('Error loading debug info:', error);
            }
        }
        
        // Cargar datos al iniciar
        async function loadData() {
            try {
                console.log('🔄 Cargando personas...');
                
                const response = await fetch('/api/persons');
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                allPersons = await response.json();
                
                console.log(`✅ Cargadas ${Object.keys(allPersons).length} personas:`, allPersons);
                
                await updateStats();
                renderPersons();
                await loadDebugInfo();
            } catch (error) {
                console.error('❌ Error loading data:', error);
                document.getElementById('persons-container').innerHTML = `
                    <div class="error-message">
                        <h2>❌ Error al cargar datos</h2>
                        <p>${error.message}</p>
                        <p>Verifica que el archivo <code>persons_data.json</code> existe y tiene datos.</p>
                        <button class="refresh-btn" onclick="forceReload()">Reintentar</button>
                    </div>
                `;
            }
        }
        
        // Actualizar estadísticas
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('total-persons').textContent = stats.total_persons;
                document.getElementById('suspicious-count').textContent = stats.suspicious_count;
                document.getElementById('normal-count').textContent = stats.normal_count;
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        // Renderizar personas
        function renderPersons() {
            const container = document.getElementById('persons-container');
            
            // Filtrar según selección
            let filtered = Object.entries(allPersons);
            
            if (currentFilter === 'suspicious') {
                filtered = filtered.filter(([_, data]) => data.is_suspicious);
            } else if (currentFilter === 'normal') {
                filtered = filtered.filter(([_, data]) => !data.is_suspicious);
            }
            
            console.log(`📋 Mostrando ${filtered.length} personas (filtro: ${currentFilter})`);
            
            if (filtered.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <h2>No hay personas en esta categoría</h2>
                        <p>Total de personas detectadas: ${Object.keys(allPersons).length}</p>
                    </div>
                `;
                return;
            }
            
            // Ordenar por timestamp (más recientes primero)
            filtered.sort((a, b) => b[1].timestamp - a[1].timestamp);
            
            container.innerHTML = '<div class="persons-grid"></div>';
            const grid = container.querySelector('.persons-grid');
            
            filtered.forEach(([personId, data]) => {
                const card = createPersonCard(personId, data);
                grid.appendChild(card);
            });
        }
        
        // Crear tarjeta de persona
        function createPersonCard(personId, data) {
            const card = document.createElement('div');
            card.className = `person-card ${data.is_suspicious ? 'suspicious' : ''}`;
            
            // Extraer solo el nombre del archivo
            const imagePath = data.image_path.replace(/\\\\/g, '/').split('/').pop();
            const imageUrl = `/images/${imagePath}`;
            
            console.log(`🖼️  Cargando imagen: ${imageUrl}`);
            
            card.innerHTML = `
                <div class="image-container">
                    <img src="${imageUrl}" 
                         alt="Persona ${data.track_id}"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='<div class=\\"image-error\\">❌ Imagen no disponible<br><small>${imagePath}</small></div>'">
                    ${data.is_suspicious ? '<div class="suspicious-badge">⚠️ SOSPECHOSO</div>' : ''}
                </div>
                <div class="person-info">
                    <h3>Persona #${data.track_id}</h3>
                    <p><strong>ID:</strong> ${personId}</p>
                    <p><strong>Fecha:</strong> ${data.date}</p>
                    <p><strong>Frames:</strong> ${data.frames_tracked}</p>
                    <p><strong>Duración:</strong> ${data.duration.toFixed(1)}s</p>
                    
                    <div class="checkbox-container">
                        <input type="checkbox" 
                               id="check-${personId}" 
                               ${data.is_suspicious ? 'checked' : ''}
                               onchange="toggleSuspicious('${personId}', this.checked)">
                        <label for="check-${personId}">Marcar como sospechoso</label>
                    </div>
                </div>
            `;
            
            return card;
        }
        
        // Cambiar estado de sospechoso
        async function toggleSuspicious(personId, isSuspicious) {
            try {
                console.log(`🔄 Marcando ${personId} como ${isSuspicious ? 'sospechoso' : 'normal'}...`);
                
                const response = await fetch(`/api/mark_suspicious/${personId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ is_suspicious: isSuspicious })
                });
                
                if (response.ok) {
                    allPersons[personId].is_suspicious = isSuspicious;
                    await updateStats();
                    renderPersons();
                    
                    console.log(`✅ Persona ${personId} marcada correctamente`);
                } else {
                    console.error('❌ Error en respuesta del servidor');
                }
            } catch (error) {
                console.error('❌ Error marking person:', error);
                alert('Error al marcar persona. Revisa la consola del navegador.');
            }
        }
        
        // Filtrar personas
        function filterPersons(filter) {
            currentFilter = filter;
            
            // Actualizar botones
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            renderPersons();
        }
        
        // Forzar recarga
        function forceReload() {
            console.log('🔄 Forzando recarga...');
            location.reload();
        }
        
        // Cargar datos al iniciar
        console.log('🚀 Iniciando aplicación...');
        loadData();
        
        // Recargar cada 10 segundos
        setInterval(loadData, 10000);
    </script>
</body>
</html>"""

# Guardar template HTML
with open(os.path.join(TEMPLATES_DIR, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html_template)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Servidor Web Iniciado")
    print("="*60)
    
    # Verificar archivos
    print("\n🔍 Verificando archivos...")
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            persons = json.load(f)
        print(f"✅ {DATA_FILE}: {len(persons)} personas encontradas")
        
        # Mostrar primeras 3 personas
        for i, (pid, data) in enumerate(list(persons.items())[:3]):
            print(f"   • {pid}")
            print(f"     Imagen: {data['image_path']}")
            print(f"     Existe: {os.path.exists(data['image_path'])}")
    else:
        print(f"❌ {DATA_FILE} no existe")
        print("   Ejecuta primero: python detection_system.py")
    
    if os.path.exists(DETECTED_PERSONS_DIR):
        images = os.listdir(DETECTED_PERSONS_DIR)
        print(f"✅ {DETECTED_PERSONS_DIR}/: {len(images)} imágenes")
    else:
        print(f"❌ {DETECTED_PERSONS_DIR}/ no existe")
    
    print("\n" + "="*60)
    print("📱 Abre tu navegador en: http://localhost:5000")
    print("="*60)
    print("\n💡 Funcionalidades:")
    print("   ✓ Ver todas las personas detectadas")
    print("   ✓ Marcar/desmarcar como sospechoso")
    print("   ✓ Filtrar por categoría")
    print("   ✓ Panel de debug con información del sistema")
    print("\n🔍 Debug:")
    print("   • Consola del navegador (F12) muestra logs detallados")
    print("   • Endpoint /api/debug para verificar el sistema")
    print("\n⚠️  Para detener el servidor: Ctrl+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)