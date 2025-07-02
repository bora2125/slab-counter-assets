from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
import io
import json
import csv
from datetime import datetime
import threading
import hashlib
import tempfile
import shutil
from contextlib import contextmanager

app = Flask(__name__)

# Configuración básica
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
DATABASE_FOLDER = 'database'
BACKUP_FOLDER = os.path.join(DATA_FOLDER, 'backups')
PERSISTENCE_FILE = os.path.join(DATA_FOLDER, 'slab_data.json')
PERSISTENCE_BACKUP = os.path.join(BACKUP_FOLDER, 'slab_data_backup.json')
DATABASE_FILE = os.path.join(DATABASE_FOLDER, 'detecciones_historicas.csv')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Sistema de bloqueos para evitar condiciones de carrera
_persistence_lock = threading.RLock()
_database_lock = threading.RLock()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

class BasicSlabDetector:
    def __init__(self):
        self.model_path = "best.pt"
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Carga el modelo YOLO"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Modelo YOLO cargado: {self.model_path}")
            else:
                print(f"❌ Error: No se encuentra el modelo en {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model = None
    
    def allowed_file(self, filename):
        """Verifica si el archivo es válido"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def detect_slabs(self, image_path, confidence=0.60):
        """Detecta palanquillas en la imagen"""
        if not self.model:
            return None, "Modelo YOLO no disponible"
        
        try:
            print(f"🔍 Procesando imagen: {image_path}")
            print(f"🎯 Confidence threshold: {confidence}")
            
            # Verificar que el archivo existe
            if not os.path.exists(image_path):
                return None, f"Archivo no encontrado: {image_path}"
            
            # Ejecutar detección
            results = self.model(image_path, verbose=True)
            
            detection_points = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                print(f"📊 Detecciones encontradas: {len(boxes)}")
                
                for i, conf_tensor in enumerate(boxes.conf):
                    conf = float(conf_tensor.item())
                    print(f"   Detección {i+1}: confianza = {conf:.3f}")
                    
                    if conf >= confidence:
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        detection_points.append({
                            'x': center_x,
                            'y': center_y,
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
            
            print(f"✅ Detecciones válidas (conf >= {confidence}): {len(detection_points)}")
            return detection_points, None
            
        except Exception as e:
            error_msg = f"Error procesando imagen: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def draw_detections(self, image_path, detections):
        """Dibuja las detecciones en la imagen"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Dibujar cada detección
            for i, detection in enumerate(detections):
                x, y = detection['x'], detection['y']
                conf = detection['confidence']
                
                # Dibujar punto central más grande y visible
                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # Punto rojo sólido
                cv2.circle(image, (x, y), 10, (255, 255, 255), 2)  # Borde blanco
                
                # Dibujar texto de confianza
                text = f"{i+1}: {conf:.2f}"
                cv2.putText(image, text, (x-20, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, text, (x-20, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Convertir a base64 para mostrar en web
            _, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer).decode()
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            print(f"❌ Error dibujando detecciones: {e}")
            return None

# Instancia global
detector = BasicSlabDetector()

# Esta función se definirá después de todas las funciones de persistencia

# ===== SISTEMA DE BASE DE DATOS CSV =====

def inicializar_base_datos():
    """Inicializa el archivo CSV si no existe"""
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs'])
        print(f"✅ Base de datos CSV inicializada: {DATABASE_FILE}")
    else:
        # Verificar que el archivo tenga header correcto
        with open(DATABASE_FILE, 'r', newline='', encoding='utf-8') as file:
            first_line = file.readline().strip()
            if not first_line.startswith('fecha,nombre_imagen,numero_lote,cantidad_slabs'):
                print(f"⚠️ Header CSV incorrecto, corrigiendo...")
                # Leer contenido actual
                file.seek(0)
                lines = file.readlines()
                
                # Reescribir con header correcto
                with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as write_file:
                    writer = csv.writer(write_file)
                    writer.writerow(['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs'])
                    
                    # Procesar líneas existentes
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('fecha,'):
                            parts = line.split(',')
                            if len(parts) >= 4:
                                writer.writerow(parts[:4])
                print(f"✅ Header CSV corregido: {DATABASE_FILE}")

def guardar_deteccion_historica(nombre_imagen, numero_lote, cantidad_slabs):
    """Guarda una detección en la base de datos histórica"""
    try:
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Verificar si el archivo existe, si no, inicializarlo
        if not os.path.exists(DATABASE_FILE):
            inicializar_base_datos()
        
        # Añadir nueva fila al CSV
        with open(DATABASE_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([fecha_actual, nombre_imagen, numero_lote, cantidad_slabs])
        
        print(f"📊 Detección guardada en BBDD: {nombre_imagen} - Lote {numero_lote} - {cantidad_slabs} slabs")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando en base de datos: {e}")
        return False

def leer_base_datos_historica():
    """Lee todos los registros de la base de datos histórica"""
    try:
        if not os.path.exists(DATABASE_FILE):
            inicializar_base_datos()
            return []
        
        registros = []
        with open(DATABASE_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Validar que la fila tenga todos los campos requeridos
                if all(key in row and row[key] is not None and row[key].strip() != '' 
                       for key in ['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs']):
                    # Limpiar y validar datos
                    try:
                        registro_limpio = {
                            'fecha': row['fecha'].strip(),
                            'nombre_imagen': row['nombre_imagen'].strip(),
                            'numero_lote': str(row['numero_lote']).strip(),
                            'cantidad_slabs': str(row['cantidad_slabs']).strip()
                        }
                        registros.append(registro_limpio)
                    except Exception as e:
                        print(f"⚠️ Fila ignorada por datos inválidos: {row}, Error: {e}")
                        continue
                else:
                    print(f"⚠️ Fila ignorada por campos faltantes: {row}")
        
        print(f"📚 Leídos {len(registros)} registros válidos de la base de datos")
        return registros
        
    except Exception as e:
        print(f"❌ Error leyendo base de datos: {e}")
        return []

def actualizar_registro_historico(fecha_original, campo, nuevo_valor):
    """Actualiza un registro específico en la base de datos histórica"""
    try:
        if not os.path.exists(DATABASE_FILE):
            return False, "Base de datos no existe"
        
        # Leer todos los registros
        registros = []
        with open(DATABASE_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                registros.append(row)
        
        # Buscar y actualizar el registro específico
        registro_encontrado = False
        for registro in registros:
            if registro['fecha'] == fecha_original:
                if campo in registro:
                    registro[campo] = str(nuevo_valor)
                    registro_encontrado = True
                    break
        
        if not registro_encontrado:
            return False, "Registro no encontrado"
        
        # Escribir todos los registros de vuelta al archivo
        with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(registros)
        
        print(f"📊 Registro actualizado: {campo} = {nuevo_valor} para fecha {fecha_original}")
        return True, f"Campo {campo} actualizado exitosamente"
        
    except Exception as e:
        print(f"❌ Error actualizando registro: {e}")
        return False, str(e)

def eliminar_registro_historico(fecha_original):
    """Elimina un registro específico de la base de datos histórica"""
    try:
        if not os.path.exists(DATABASE_FILE):
            return False, "Base de datos no existe"
        
        # Leer todos los registros
        registros = []
        with open(DATABASE_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['fecha'] != fecha_original:
                    registros.append(row)
        
        # Escribir registros filtrados de vuelta al archivo
        with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(registros)
        
        print(f"📊 Registro eliminado del histórico: fecha {fecha_original}")
        return True, "Registro eliminado exitosamente"
        
    except Exception as e:
        print(f"❌ Error eliminando registro: {e}")
        return False, str(e)

def sincronizar_lote_con_historico(nombre_imagen, numero_lote_anterior, numero_lote_nuevo, cantidad_slabs):
    """Sincroniza cambios de lote con el histórico - maneja reorganización inteligente"""
    try:
        # Buscar el registro más reciente del lote anterior para esta imagen
        registros = leer_base_datos_historica()
        registro_anterior = None
        for registro in registros:
            if (registro['nombre_imagen'] == nombre_imagen and 
                str(registro['numero_lote']) == str(numero_lote_anterior)):
                # Guardar el más reciente (último en la lista)
                registro_anterior = registro
        
        if registro_anterior:
            cantidad_anterior = int(registro_anterior['cantidad_slabs'])
            
            if numero_lote_anterior == numero_lote_nuevo:
                # Mismo lote, solo actualizar cantidad (reorganización parcial)
                print(f"📊 Actualizando cantidad del lote {numero_lote_anterior}: {cantidad_anterior} → {cantidad_slabs}")
                actualizar_registro_historico(registro_anterior['fecha'], 'cantidad_slabs', cantidad_slabs)
            else:
                # Diferente lote: cambio completo de número
                print(f"📊 Cambiando lote completo: {numero_lote_anterior} → {numero_lote_nuevo}")
                actualizar_registro_historico(registro_anterior['fecha'], 'numero_lote', numero_lote_nuevo)
                actualizar_registro_historico(registro_anterior['fecha'], 'cantidad_slabs', cantidad_slabs)
        else:
            print(f"⚠️ No se encontró registro anterior para lote {numero_lote_anterior} en {nombre_imagen}")
        
        print(f"📊 Lote sincronizado: {nombre_imagen} - {numero_lote_anterior} → {numero_lote_nuevo}")
        return True
        
    except Exception as e:
        print(f"❌ Error sincronizando lote: {e}")
        return False

# ===== SISTEMA DE PERSISTENCIA =====

@contextmanager
def persistence_file_lock():
    """Context manager para bloqueo seguro del archivo de persistencia"""
    with _persistence_lock:
        yield

def calculate_file_hash(filepath):
    """Calcula hash MD5 de un archivo para verificar integridad"""
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def create_backup_if_needed():
    """Crea respaldo del archivo de persistencia si es necesario"""
    try:
        if os.path.exists(PERSISTENCE_FILE):
            # Verificar si necesita respaldo (cada hora)
            backup_needed = True
            if os.path.exists(PERSISTENCE_BACKUP):
                persistence_time = os.path.getmtime(PERSISTENCE_FILE)
                backup_time = os.path.getmtime(PERSISTENCE_BACKUP)
                backup_needed = (persistence_time - backup_time) > 3600  # 1 hora
            
            if backup_needed:
                shutil.copy2(PERSISTENCE_FILE, PERSISTENCE_BACKUP)
                print(f"💾 Respaldo creado: {PERSISTENCE_BACKUP}")
    except Exception as e:
        print(f"⚠️ Error creando respaldo: {e}")

def load_persistent_data():
    """Carga datos persistentes desde archivo JSON con validación robusta"""
    with persistence_file_lock():
        # Intentar cargar archivo principal
        for attempt_file in [PERSISTENCE_FILE, PERSISTENCE_BACKUP]:
            if not os.path.exists(attempt_file):
                continue
                
            try:
                with open(attempt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validar estructura básica
                if not isinstance(data, dict):
                    raise ValueError("Datos no tienen estructura dict")
                
                if 'images' not in data:
                    data['images'] = []
                
                # Validar cada imagen
                valid_images = []
                for img in data.get('images', []):
                    if isinstance(img, dict) and 'name' in img:
                        # Asegurar campos requeridos
                        img.setdefault('status', 'loaded')
                        img.setdefault('manualPoints', [])
                        img.setdefault('batches', [])
                        img.setdefault('nextPointId', 1)
                        valid_images.append(img)
                
                data['images'] = valid_images
                data.setdefault('next_image_id', 1)
                data.setdefault('last_updated', None)
                
                # Si es el respaldo, restaurar al principal
                if attempt_file == PERSISTENCE_BACKUP:
                    print(f"🔄 Restaurando desde respaldo...")
                    save_persistent_data_internal(data)
                
                print(f"✅ Datos cargados: {len(data.get('images', []))} imágenes")
                return data
                
            except Exception as e:
                print(f"❌ Error con {attempt_file}: {e}")
                continue
        
        # Si no se pudo cargar ningún archivo, crear estructura nueva
        print("🆕 Creando estructura de datos nueva")
        return {"images": [], "next_image_id": 1, "last_updated": None}

def save_persistent_data_internal(data):
    """Función interna para guardar sin bloqueo (ya debe estar en contexto de bloqueo)"""
    try:
        data["last_updated"] = datetime.now().isoformat()
        
        # Escribir a archivo temporal primero (operación atómica)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                         dir=DATA_FOLDER, delete=False) as temp_file:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_filename = temp_file.name
        
        # Verificar que se escribió correctamente
        try:
            with open(temp_filename, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            if len(test_data.get('images', [])) != len(data.get('images', [])):
                raise ValueError("Verificación de integridad falló")
        except Exception as e:
            os.unlink(temp_filename)
            raise e
        
        # Mover archivo temporal al definitivo (operación atómica)
        shutil.move(temp_filename, PERSISTENCE_FILE)
        
        print(f"💾 Datos guardados: {len(data.get('images', []))} imágenes")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando datos: {e}")
        return False

def save_persistent_data(data):
    """Guarda datos persistentes en archivo JSON de forma segura"""
    with persistence_file_lock():
        create_backup_if_needed()
        return save_persistent_data_internal(data)

def find_image_data_by_name(filename):
    """Busca datos de imagen por nombre de archivo de forma segura"""
    with persistence_file_lock():
        persistent_data = load_persistent_data()
        for img_data in persistent_data.get('images', []):
            if img_data.get('name') == filename:
                return img_data.copy()  # Retornar copia para evitar modificaciones accidentales
        return None

def verify_image_exists_and_has_data(filename):
    """Verifica si una imagen existe y tiene datos de lotes"""
    img_data = find_image_data_by_name(filename)
    if img_data is None:
        return False, "Imagen no encontrada en datos persistentes"
    
    status = img_data.get('status', 'loaded')
    manual_points = img_data.get('manualPoints', [])
    batches = img_data.get('batches', [])
    
    if status == 'with-batches' and (manual_points or batches):
        return True, f"Imagen con {len(manual_points)} puntos y {len(batches)} lotes"
    else:
        return False, f"Imagen en estado '{status}' sin datos de lotes"

def save_image_data(image_data):
    """Guarda o actualiza datos de una imagen específica de forma robusta"""
    if not image_data or not image_data.get('name'):
        print("❌ Error: Datos de imagen inválidos")
        return False
    
    with persistence_file_lock():
        persistent_data = load_persistent_data()
        
        # Buscar si ya existe
        existing_index = -1
        for i, img_data in enumerate(persistent_data.get('images', [])):
            if img_data.get('name') == image_data.get('name'):
                existing_index = i
                break
        
        # Preparar datos OPTIMIZADOS para guardar (solo lo esencial)
        detection_summary = None
        if image_data.get('detectionData'):
            # Solo guardar resumen de detección, NO los datos completos pesados
            detection_summary = {
                'count': image_data['detectionData'].get('count', 0),
                'confidence_used': image_data.get('confidence_used', 0.60),
                'detected_at': datetime.now().isoformat()
            }
        
        # Preservar datos existentes importantes
        existing_data = {}
        if existing_index >= 0:
            existing_data = persistent_data['images'][existing_index]
        
        data_to_save = {
            'name': image_data.get('name'),
            'status': image_data.get('status', existing_data.get('status', 'loaded')),
            'manualPoints': image_data.get('manualPoints', existing_data.get('manualPoints', [])),
            'batches': image_data.get('batches', existing_data.get('batches', [])),
            'nextPointId': image_data.get('nextPointId', existing_data.get('nextPointId', 1)),
            'detectionSummary': detection_summary or existing_data.get('detectionSummary'),
            'createdAt': existing_data.get('createdAt', datetime.now().isoformat()),
            'updatedAt': datetime.now().isoformat()
        }
        
        # Validar datos antes de guardar
        if not isinstance(data_to_save['manualPoints'], list):
            data_to_save['manualPoints'] = []
        if not isinstance(data_to_save['batches'], list):
            data_to_save['batches'] = []
        
        if existing_index >= 0:
            # Actualizar existente
            persistent_data['images'][existing_index] = data_to_save
            print(f"🔄 Datos actualizados para: {image_data.get('name')}")
        else:
            # Agregar nuevo
            persistent_data['images'].append(data_to_save)
            print(f"➕ Nuevos datos guardados para: {image_data.get('name')}")
        
        # Sincronizar con base de datos CSV si tiene lotes
        if data_to_save['status'] == 'with-batches' and data_to_save['batches']:
            sync_with_database(data_to_save)
        
        return save_persistent_data_internal(persistent_data)

def sync_with_database(image_data):
    """Sincroniza datos de imagen con la base de datos CSV"""
    try:
        if not image_data.get('batches'):
            return
        
        with _database_lock:
            # Verificar si ya existe en CSV
            existing_records = leer_base_datos_historica()
            image_name = image_data['name']
            
            # Remover registros antiguos de esta imagen
            filtered_records = [r for r in existing_records if r['nombre_imagen'] != image_name]
            
            # Agregar registros actuales
            for batch in image_data['batches']:
                batch_number = batch.get('number', 'N/A')
                slab_count = len([p for p in image_data.get('manualPoints', []) 
                                if p.get('batchNumber') == batch_number])
                
                new_record = {
                    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'nombre_imagen': image_name,
                    'numero_lote': str(batch_number),
                    'cantidad_slabs': str(slab_count)
                }
                filtered_records.append(new_record)
            
            # Escribir de vuelta al CSV
            escribir_base_datos_historica(filtered_records)
            
    except Exception as e:
        print(f"⚠️ Error sincronizando con CSV: {e}")

def optimize_persistent_data():
    """Optimiza el archivo de persistencia eliminando datos pesados innecesarios"""
    with persistence_file_lock():
        try:
            data = load_persistent_data()
            
            # Optimizar cada imagen
            optimized_images = []
            for img_data in data.get('images', []):
                # Crear versión optimizada sin detectionData pesado
                optimized_img = {
                    'name': img_data.get('name'),
                    'status': img_data.get('status'),
                    'manualPoints': img_data.get('manualPoints', []),
                    'batches': img_data.get('batches', []),
                    'nextPointId': img_data.get('nextPointId', 1),
                    'detectionSummary': img_data.get('detectionSummary'),
                    'createdAt': img_data.get('createdAt'),
                    'updatedAt': img_data.get('updatedAt')
                }
                optimized_images.append(optimized_img)
            
            # Guardar versión optimizada
            optimized_data = {
                'images': optimized_images,
                'next_image_id': data.get('next_image_id', 1),
                'last_updated': datetime.now().isoformat(),
                'optimized_at': datetime.now().isoformat()
            }
            
            success = save_persistent_data_internal(optimized_data)
            if success:
                print(f"✅ Archivo optimizado: {len(optimized_images)} imágenes")
            return success
            
        except Exception as e:
            print(f"❌ Error optimizando persistencia: {e}")
            return False

def escribir_base_datos_historica(registros):
    """Escribe registros a la base de datos CSV de forma segura"""
    try:
        with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['fecha', 'nombre_imagen', 'numero_lote', 'cantidad_slabs']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(registros)
    except Exception as e:
        print(f"❌ Error escribiendo CSV: {e}")
        raise e

@app.route('/')
def index():
    """Página principal"""
    return render_template('basic_index_v8.html')

@app.route('/logo_aza.JPG')
def serve_logo():
    """Servir el logo de AZA"""
    return send_from_directory('.', 'logo_aza.JPG')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Sube archivo"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and detector.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📁 Archivo guardado: {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect', methods=['POST'])
def detect():
    """Ejecuta detección"""
    data = request.get_json()
    filepath = data.get('filepath')
    confidence = float(data.get('confidence', 0.60))
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
    
    print(f"\n🚀 Iniciando detección...")
    print(f"📂 Archivo: {filepath}")
    print(f"🎯 Confidence: {confidence}")
    
    # Detectar palanquillas
    detections, error = detector.detect_slabs(filepath, confidence)
    
    if error:
        return jsonify({'error': error}), 500
    
    # Dibujar detecciones
    image_with_detections = detector.draw_detections(filepath, detections)
    
    if not image_with_detections:
        return jsonify({'error': 'Error generating result image'}), 500
    
    result = {
        'success': True,
        'count': len(detections),
        'detections': detections,
        'image_data': image_with_detections
    }
    
    print(f"✅ Resultado: {len(detections)} palanquillas detectadas")
    return jsonify(result)

@app.route('/load_persistent_data', methods=['GET'])
def load_data_route():
    """Endpoint para cargar datos persistentes"""
    try:
        persistent_data = load_persistent_data()
        return jsonify({
            'success': True,
            'data': persistent_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/save_image_data', methods=['POST'])
def save_image_data_route():
    """Endpoint para guardar datos de imagen"""
    try:
        data = request.get_json()
        image_data = data.get('imageData')
        
        if not image_data or not image_data.get('name'):
            return jsonify({
                'success': False,
                'error': 'Datos de imagen inválidos'
            }), 400
        
        success = save_image_data(image_data)
        
        return jsonify({
            'success': success,
            'message': f"Datos guardados para: {image_data.get('name')}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_image_data/<filename>', methods=['GET'])
def get_image_data_route(filename):
    """Endpoint para obtener datos de una imagen específica"""
    try:
        image_data = find_image_data_by_name(filename)
        
        if image_data:
            return jsonify({
                'success': True,
                'data': image_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No se encontraron datos para esta imagen'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/optimize_data', methods=['POST'])
def optimize_data_route():
    """Endpoint para optimizar el archivo de persistencia"""
    try:
        success = optimize_persistent_data()
        if success:
            return jsonify({
                'success': True,
                'message': 'Archivo de persistencia optimizado exitosamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error optimizando archivo'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===== RUTAS DE BASE DE DATOS HISTÓRICA =====

@app.route('/guardar_lote_historico', methods=['POST'])
def guardar_lote_historico():
    """Endpoint para guardar un lote en la base de datos histórica"""
    try:
        data = request.get_json()
        nombre_imagen = data.get('nombre_imagen')
        numero_lote = data.get('numero_lote')
        cantidad_slabs = data.get('cantidad_slabs')
        
        if not all([nombre_imagen, numero_lote is not None, cantidad_slabs is not None]):
            return jsonify({
                'success': False,
                'error': 'Faltan datos requeridos'
            }), 400
        
        success = guardar_deteccion_historica(nombre_imagen, numero_lote, cantidad_slabs)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Lote {numero_lote} guardado en base de datos histórica'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error guardando en base de datos'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/obtener_datos_historicos', methods=['GET'])
def obtener_datos_historicos():
    """Endpoint para obtener todos los datos históricos"""
    try:
        registros = leer_base_datos_historica()
        return jsonify({
            'success': True,
            'data': registros,
            'total': len(registros)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/actualizar_registro_historico', methods=['POST'])
def actualizar_registro_historico_route():
    """Endpoint para actualizar un registro específico en la base de datos histórica"""
    try:
        data = request.get_json()
        fecha_original = data.get('fecha_original')
        campo = data.get('campo')
        nuevo_valor = data.get('nuevo_valor')
        
        if not all([fecha_original, campo, nuevo_valor is not None]):
            return jsonify({
                'success': False,
                'error': 'Faltan datos requeridos: fecha_original, campo, nuevo_valor'
            }), 400
        
        # Validar campo permitido
        campos_permitidos = ['nombre_imagen', 'numero_lote', 'cantidad_slabs']
        if campo not in campos_permitidos:
            return jsonify({
                'success': False,
                'error': f'Campo no permitido: {campo}. Campos permitidos: {campos_permitidos}'
            }), 400
        
        # Validaciones específicas por campo
        if campo in ['numero_lote', 'cantidad_slabs']:
            try:
                nuevo_valor = int(nuevo_valor)
                if nuevo_valor < 1:
                    return jsonify({
                        'success': False,
                        'error': f'{campo} debe ser un número mayor a 0'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'{campo} debe ser un número válido'
                }), 400
        
        # Actualizar registro
        success, message = actualizar_registro_historico(fecha_original, campo, nuevo_valor)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/eliminar_registro_historico', methods=['POST'])
def eliminar_registro_historico_route():
    """Endpoint para eliminar un registro del histórico"""
    try:
        data = request.get_json()
        fecha_original = data.get('fecha_original')
        
        if not fecha_original:
            return jsonify({
                'success': False,
                'error': 'Falta fecha_original'
            }), 400
        
        success, message = eliminar_registro_historico(fecha_original)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/sincronizar_lotes_historico', methods=['POST'])
def sincronizar_lotes_historico_route():
    """Endpoint para sincronizar cambios de lotes con el histórico"""
    try:
        data = request.get_json()
        nombre_imagen = data.get('nombre_imagen')
        numero_lote_anterior = data.get('numero_lote_anterior')
        numero_lote_nuevo = data.get('numero_lote_nuevo') 
        cantidad_slabs = data.get('cantidad_slabs')
        
        if not all([nombre_imagen, numero_lote_anterior is not None, 
                   numero_lote_nuevo is not None, cantidad_slabs is not None]):
            return jsonify({
                'success': False,
                'error': 'Faltan datos requeridos'
            }), 400
        
        success = sincronizar_lote_con_historico(nombre_imagen, numero_lote_anterior, 
                                               numero_lote_nuevo, cantidad_slabs)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Lotes sincronizados exitosamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error sincronizando lotes'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic Slab Detector')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    port = args.port
    
    print("\n" + "="*60)
    print("🔍 BASIC SLAB DETECTOR V8 - Nueva Generación con Base de Datos CSV")
    print("="*60)
    
    # INICIALIZAR BASE DE DATOS CSV
    print("📊 Inicializando base de datos CSV...")
    inicializar_base_datos()
    
    # OPTIMIZAR PERSISTENCIA AL INICIO
    print("🔧 Optimizando archivo de persistencia...")
    optimize_success = optimize_persistent_data()
    if optimize_success:
        print("✅ Persistencia optimizada exitosamente")
    else:
        print("⚠️ Advertencia: No se pudo optimizar la persistencia")
    print(f"📁 Modelo: {detector.model_path}")
    print(f"🤖 Estado del modelo: {'✅ Cargado' if detector.model else '❌ Error'}")
    print(f"📂 Carpeta uploads: {UPLOAD_FOLDER}")
    print("="*60)
    print("🌐 ACCESO AL SERVIDOR:")
    print(f"   Desde WSL:     http://localhost:{port}")
    print(f"   Desde Windows: http://172.20.135.105:{port}")
    print(f"   También:       http://127.0.0.1:{port}")
    print("="*60)
    print("💡 INSTRUCCIONES:")
    print("   1. Abre tu navegador en WINDOWS")
    print(f"   2. Ve a: http://172.20.135.105:{port}")
    print(f"   3. Si no funciona, prueba: http://localhost:{port}")
    print("="*60)
    print("🛑 Presiona CTRL+C para detener el servidor")
    print("="*60)
    
    # Inicializar sistema de persistencia robusta
    print("🔧 Inicializando sistema de persistencia robusto...")
    try:
        # Verificar y optimizar datos existentes
        if os.path.exists(PERSISTENCE_FILE):
            file_size = os.path.getsize(PERSISTENCE_FILE)
            if file_size > 500000:  # Si es mayor a 500KB, optimizar
                print(f"⚠️ Archivo grande detectado ({file_size} bytes), optimizando...")
                optimize_persistent_data()
        
        # Crear respaldo inicial
        create_backup_if_needed()
        
        # Verificar integridad de datos
        test_data = load_persistent_data()
        print(f"✅ Sistema de persistencia inicializado: {len(test_data.get('images', []))} imágenes")
    except Exception as e:
        print(f"⚠️ Error inicializando persistencia: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=True)