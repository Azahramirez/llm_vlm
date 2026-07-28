import ollama
import time
import os
import cv2
import json
import numpy as np
import re

model_name = "qwen3.5:9b"  # "qwen2.5vl:3b" "qwen3.5:0.8b"

# Deterministic settings — precise coordinates want greedy decoding, not sampling
params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 300,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop": None,
}

print(f"Modelo de Visión Seleccionado: {model_name}\n")

def preguntar(pregunta, imagen=None):
    start_time = time.time()
    messages = [{'role': 'user', 'content': pregunta}]
    if imagen:
        if not os.path.exists(imagen):
            print(f"❌ Error: No se encontró la imagen en la ruta '{imagen}'")
            return "[]"
        messages[0]['images'] = [imagen]
    response = ollama.chat(model=model_name, messages=messages, options=params)
    print(f"\n⏱ Tiempo de inferencia: {time.time() - start_time:.2f} segundos")
    return response['message']['content']

image_path = "my_image.jpg"

# Format now matches what Qwen2.5-VL is actually trained to output: [x1,y1,x2,y2]
prompt = """
Detect all distinct objects in the image.

Return ONLY a valid JSON array, nothing else (no markdown fences, no explanation).
Each element must be an object with exactly these two keys:
- "label": a short string describing the object
- "bbox_2d": [x1, y1, x2, y2] in pixel coordinates of the image,
  where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

If you are unsure about an object, omit it rather than guessing.

Example of the exact format required:
[{"label": "dog", "bbox_2d": [34, 120, 210, 400]}]
"""
respuesta = preguntar(prompt, imagen=image_path)

print("\n--- Respuesta cruda del VLM ---")
print(respuesta)
print("-------------------------------\n")

def parse_vlm_string(vlm_string):
    cleaned = re.sub(r"^```json\s*", "", vlm_string.strip())
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("[")
    if start == -1:
        print("No se encontró un array JSON en la respuesta.")
        return []

    depth, end = 0, -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        candidate = cleaned[start:end + 1]
    else:
        last_obj_end = cleaned.rfind("}")
        if last_obj_end == -1:
            return []
        candidate = cleaned[start:last_obj_end + 1] + "]"

    try:
        data = json.loads(candidate)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"Failed parsing string. Error: {e}")
        return []

detections = parse_vlm_string(respuesta)
print(f"Detecciones parseadas: {len(detections)}")

try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("No se pudo cargar la imagen original. Generando lienzo negro alternativo.")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

h_img, w_img, _ = img.shape
image_resized = img.copy()

COLOR_MAP = {"person": (0, 255, 0), "forklift": (255, 0, 0), "barrel": (0, 165, 255)}
DEFAULT_COLOR = (0, 255, 0)

drawn = 0
for obj in detections:
    bbox = obj.get('bbox_2d')
    label = obj.get('label', 'object')
    if not bbox or len(bbox) != 4:
        continue

    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = sorted([int(x_min), int(x_max)])
    y_min, y_max = sorted([int(y_min), int(y_max)])
    x_min, x_max = max(0, min(x_min, w_img - 1)), max(0, min(x_max, w_img - 1))
    y_min, y_max = max(0, min(y_min, h_img - 1)), max(0, min(y_max, h_img - 1))
    if x_max <= x_min or y_max <= y_min:
        continue

    color = COLOR_MAP.get(label.lower(), DEFAULT_COLOR)
    cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image_resized, label, (x_min, max(0, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    drawn += 1

print(f"Cajas dibujadas: {drawn}")

model_name_safe = model_name.replace(":", "_").replace("/", "_")
image_path_safe = image_path.replace(":", "_").replace("/", "_")
output_name = f"resultadoOLLAMA_{model_name_safe}_{image_path_safe}.jpg"
cv2.imwrite(output_name, image_resized)
print(f"✅ Imagen guardada con éxito como: '{output_name}'")

try:
    cv2.imshow("VLM Target Visualization", image_resized)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass