# web_interface.py
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
    return render_template('index.html')

@app.route('/api/persons')
def get_persons():
    """Obtiene lista de todas las personas detectadas"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            persons = json.load(f)
        return jsonify(persons)
    return jsonify({})

@app.route('/api/mark_suspicious/<person_id>', methods=['POST'])
def mark_suspicious(person_id):
    """Marca una persona como sospechosa"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            persons = json.load(f)
        
        if person_id in persons:
            data = request.json
            persons[person_id]['is_suspicious'] = data.get('is_suspicious', True)
            
            with open(DATA_FILE, 'w') as f:
                json.dump(persons, f, indent=2)
            
            return jsonify({'success': True, 'person_id': person_id})
    
    return jsonify({'success': False, 'error': 'Person not found'}), 404

@app.route('/api/stats')
def get_stats():
    """Obtiene estadísticas generales"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            persons = json.load(f)
        
        total = len(persons)
        suspicious = sum(1 for p in persons.values() if p.get('is_suspicious', False))
        
        return jsonify({
            'total_persons': total,
            'suspicious_count': suspicious,
            'normal_count': total - suspicious
        })
    
    return jsonify({'total_persons': 0, 'suspicious_count': 0, 'normal_count': 0})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sirve las imágenes de personas detectadas"""
    return send_from_directory(DETECTED_PERSONS_DIR, filename)

# =======================
# GENERAR HTML
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
        }
        
        .person-card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 Sistema de Gestión de Personas Detectadas</h1>
            <p style="color: #666; margin-top: 10px;">Revisa y marca personas sospechosas</p>
        </header>
        
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
        </div>
        
        <div id="persons-container" class="loading">
            Cargando personas detectadas...
        </div>
    </div>

    <script>
        let allPersons = {};
        let currentFilter = 'all';
        
        // Cargar datos al iniciar
        async function loadData() {
            try {
                const response = await fetch('/api/persons');
                allPersons = await response.json();
                
                await updateStats();
                renderPersons();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('persons-container').innerHTML = 
                    '<div class="empty-state"><h2>Error al cargar datos</h2></div>';
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
            
            if (filtered.length === 0) {
                container.innerHTML = '<div class="empty-state"><h2>No hay personas en esta categoría</h2></div>';
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
            
            const imagePath = data.image_path.replace(/\\/g, '/').split('/').pop();
            
            card.innerHTML = `
                <div class="image-container">
                    <img src="/images/${imagePath}" alt="Persona ${data.track_id}">
                    ${data.is_suspicious ? '<div class="suspicious-badge">⚠️ SOSPECHOSO</div>' : ''}
                </div>
                <div class="person-info">
                    <h3>Persona #${data.track_id}</h3>
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
                    
                    console.log(`✅ Persona ${personId} marcada como ${isSuspicious ? 'sospechosa' : 'normal'}`);
                }
            } catch (error) {
                console.error('Error marking person:', error);
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
        
        // Cargar datos al iniciar
        loadData();
        
        // Recargar cada 5 segundos
        setInterval(loadData, 5000);
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
    print("📱 Abre tu navegador en: http://localhost:5000")
    print("="*60)
    print("\n💡 Funcionalidades:")
    print("   ✓ Ver todas las personas detectadas")
    print("   ✓ Marcar/desmarcar como sospechoso (checkbox)")
    print("   ✓ Filtrar por categoría (Todas/Sospechosas/Normales)")
    print("   ✓ Actualización automática cada 5 segundos")
    print("\n⚠️  Para detener el servidor: Ctrl+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)