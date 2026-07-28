import json
import os
import time
import re
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

# 1. Modelo
model_id = "Qwen/Qwen3.5-0.8B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando modelo {model_id} en el dispositivo: {device.upper()}")

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map={"": device},
)
processor = AutoProcessor.from_pretrained(model_id)

hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
local_path = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
print(f"Model and processor local storage directory: {os.path.abspath(local_path)}")

# 2. Imagen
image_path = "huskys2.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

image = Image.open(image_path).convert("RGB")
img_w, img_h = image.size

# 3. Prompt — pide explícitamente el formato NORMALIZADO 0-1000 que
# Qwen3.5/Qwen3-VL realmente usa (a diferencia de Qwen2.5-VL, que usaba
# píxeles absolutos). Pedir el formato equivocado es la causa más común
# de bounding boxes desalineadas. # distinct objects in the image.
prompt_text = """Detect all distinct huskies in the image.

Return ONLY a valid JSON array, nothing else (no markdown fences, no explanation).
Each element must be an object with exactly these two keys:
- "label": a short string describing the object
- "bbox_2d": [x1, y1, x2, y2] normalized to a 0-1000 scale relative to the image
  width and height, where (x1, y1) is the top-left corner and (x2, y2) is the
  bottom-right corner.

If you are unsure about an object, omit it rather than guessing.

Example of the exact format required (separate each husky as a different label):
[{"label": "husky1", "bbox_2d": [120, 340, 610, 900]}]
[{"label": "husky2", "bbox_2d": [120, 340, 610, 900]}]
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# 4. Tokenización / preparación de inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# 5. Inferencia
print(f"\nAnalizando imagen en {device.upper()}...")
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

inference_time = time.time() - start_time

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\n--- Qwen3.5-0.8B Output ---")
print(output_text)
print("\n--- Performance Metrics ---")
print(f"Inference execution time: {inference_time:.4f} seconds")
print("-------------------------------\n")


def parse_vlm_string(vlm_string):
    """Extrae y repara el array JSON de la respuesta del modelo."""
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


detections = parse_vlm_string(output_text)
print(f"Detecciones parseadas: {len(detections)}")

# ================== CARGA DE IMAGEN PARA DIBUJAR ==================
img = cv2.imread(image_path)
if img is None:
    print("No se pudo cargar la imagen original. Generando lienzo negro alternativo.")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

h_img, w_img, _ = img.shape
image_resized = img.copy()

COLOR_MAP = {
    "person": (0, 255, 0),
    "forklift": (255, 0, 0),
    "barrel": (0, 165, 255),
}
DEFAULT_COLOR = (0, 255, 0)

drawn = 0
for obj in detections:
    bbox = obj.get("bbox_2d")
    label = obj.get("label", "object")
    if not bbox or len(bbox) != 4:
        continue

    # Conversión CLAVE: bbox_2d viene normalizado 0-1000, hay que
    # escalarlo al tamaño real de la imagen antes de dibujar.
    x1, y1, x2, y2 = bbox
    x_min = int(x1 / 1000 * w_img)
    y_min = int(y1 / 1000 * h_img)
    x_max = int(x2 / 1000 * w_img)
    y_max = int(y2 / 1000 * h_img)

    x_min, x_max = sorted([x_min, x_max])
    y_min, y_max = sorted([y_min, y_max])
    x_min = max(0, min(x_min, w_img - 1))
    x_max = max(0, min(x_max, w_img - 1))
    y_min = max(0, min(y_min, h_img - 1))
    y_max = max(0, min(y_max, h_img - 1))

    if x_max <= x_min or y_max <= y_min:
        continue

    color = COLOR_MAP.get(label.lower(), DEFAULT_COLOR)
    cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image_resized, label, (x_min, max(0, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    drawn += 1

print(f"Cajas dibujadas: {drawn}")

# ================== GUARDAR RESULTADO ==================
model_name_safe = model_id.replace(":", "_").replace("/", "_")
image_path_safe = image_path.replace(":", "_").replace("/", "_")
output_name = f"resultado_{model_name_safe}_{image_path_safe}.jpg"
cv2.imwrite(output_name, image_resized)
print(f"✅ Imagen guardada con éxito como: '{output_name}'")

try:
    cv2.imshow("VLM Target Visualization", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass