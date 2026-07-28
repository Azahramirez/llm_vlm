import ollama
import time
import os
import cv2
import json
import numpy as np
import re

# ================== CONFIGURACIÓN CORREGIDA ==================
# DEBES usar un modelo "vision". Asegúrate de correr en tu terminal:
# ollama pull qwen3.5-vision
model_name =  "qwen3.5:0.8b" #"qwen2.5vl:3b" "qwen3.5:0.8b"

# Parámetros optimizados para detección de objetos (BBoxes)
params = {
    "temperature": 0.5,       # 0.0 elimina la aleatoriedad; vital para coordenadas exactas
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 1600,       # Aumentado! 100 tokens cortaba el JSON a la mitad
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop": None,
}

print(f"Modelo de Visión Seleccionado: {model_name}\n")

# ================== FUNCIÓN PARA CHAT ==================
def preguntar(pregunta, imagen=None):
    start_time = time.time()
    
    messages = [{'role': 'user', 'content': pregunta}]
    
    if imagen:
        if not os.path.exists(imagen):
            print(f"❌ Error: No se encontró la imagen en la ruta '{imagen}'")
            return "[]"
        messages[0]['images'] = [imagen]
    
    response = ollama.chat(
        model=model_name,
        messages=messages,
        #options=params
    )
    
    end_time = time.time()
    tiempo = end_time - start_time
    
    print(f"\n⏱ Tiempo de inferencia: {tiempo:.2f} segundos")
    return response['message']['content']

# ================== EJECUCIÓN ==================
image_path = "my_image.jpg" 

prompt = """
Generate a json file with the bounding box objects and their labels in the image. Use the format ['bbox_2d': [xmin, ymin, xmax, ymax]}, ...]. 

Reason about your detections and provide the most accurate bounding boxes possible. If you are unsure about an object, do not include it in the output, dont add text detected.

"""
#prompt="Hello"
respuesta = preguntar(prompt, imagen=image_path)

print("\n--- Respuesta cruda del VLM ---")
print(respuesta)
print("-------------------------------\n")

def parse_vlm_string(vlm_string):
    """Extrae, repara y parsea el contenido JSON de la respuesta."""
    cleaned = re.sub(r"^```json\s*", "", vlm_string.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Si el modelo aún así se corta, cerramos el JSON limpiamente
    last_valid_object_idx = cleaned.rfind("}")
    if last_valid_object_idx != -1:
        cleaned = cleaned[: last_valid_object_idx + 1] + "\n]"
    else:
        cleaned = "[]"

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed parsing string. Error: {e}")
        return []

detections = parse_vlm_string(respuesta)



# Configuración de OpenCV
COLOR_MAP = {
    "person": (0, 255, 0),    # Verde
    "forklift": (255, 0, 0),  # Azul
    "barrel": (0, 165, 255),  # Naranja
}

try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("No se pudo cargar la imagen original para dibujar. Generando lienzo negro alternativo.")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

h_img, w_img, _ = img.shape

# resize image
image_resized = cv2.resize(img, (w_img, h_img))

# ================== DIBUJAR CAJAS ==================
bbox_data=detections
# Draw bounding boxes
for obj in bbox_data:
    x_min, y_min, x_max, y_max = obj['bbox_2d']
    label = obj['label']
    
    # Define color (Green: BGR 0, 255, 0)
    color = (0, 255, 0)
    
    # Draw rectangle
    cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw label text
    # Position: just above the top-left corner of the box
    cv2.putText(image_resized, label, (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ================== GUARDAR RESULTADO ==================
model_name_safe = model_name.replace(":", "_").replace("/", "_")
image_path_safe = image_path.replace(":", "_").replace("/", "_")
output_name = f"resultadoOLMA_{model_name_safe}_{image_path_safe}.jpg"
cv2.imwrite(output_name, image_resized)

print(f"✅ Imagen guardada con éxito como: '{output_name}'")

# Mostrar ventana si tienes entorno gráfico activo
try:
    cv2.imshow("VLM Target Visualization", image_resized)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass