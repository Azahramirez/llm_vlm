import json
import os
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import cv2
import re
import numpy as np

# 1. Identificador del modelo (Versión 3B)
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# 2. Configuración estricta para CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando modelo {model_id} en el dispositivo: {device.upper()}")

# 3. Carga del modelo usando la clase arquitectónica correcta
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)
processor = AutoProcessor.from_pretrained(model_id)

# --- Mostrar el directorio de almacenamiento local ---
hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
local_path = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
print(f"Model and processor local storage directory: {os.path.abspath(local_path)}")

# 4. Preparación de la imagen
local_image_path = "huskys.jpg"
image_path = local_image_path
if not os.path.exists(local_image_path):
    raise FileNotFoundError(f"No se encontró la imagen en: {local_image_path}")

image = Image.open(local_image_path).convert("RGB")

# 5. Construcción del payload
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Generate a json file with the bounding box objects and their labels in the image. Use the format ['bbox_2d': [ymin, xmin, ymax, xmax]}}, ...]. Reason about your detections and provide the most accurate bounding boxes possible. If you are unsure about an object, do not include it in the output, dont add text detected."}
        ]
    }
]

text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 6. Procesamiento de tensores
inputs = processor(
    text=[text],
    images=image,
    videos=None,
    padding=True,
    return_tensors="pt"
).to(device)

# 7. Inferencia
print("\nAnalizando imagen en CPU...")

# --- Iniciar temporizador de inferencia ---
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# --- Finalizar temporizador de inferencia ---
inference_time = time.time() - start_time

# 8. Decodificación de salida
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text_i = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print("\n--- Qwen2.5-VL 3B Output ---")
print(output_text_i)

# --- NUEVO: Imprimir el tiempo de ejecución calculado ---
print("\n--- Performance Metrics ---")
print(f"Inference execution time: {inference_time:.4f} seconds")

respuesta = output_text_i


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
model_name="Qwen2.5-VL 3B"
model_name_safe = model_name.replace(":", "_").replace("/", "_")
image_path_safe = image_path.replace(":", "_").replace("/", "_")
output_name = f"resultado_{model_name_safe}_{image_path_safe}.jpg"
cv2.imwrite(output_name, image_resized)

print(f"✅ Imagen guardada con éxito como: '{output_name}'")

# Mostrar ventana si tienes entorno gráfico activo
try:
    cv2.imshow("VLM Target Visualization", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass