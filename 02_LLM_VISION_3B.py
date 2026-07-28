import os
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
local_image_path = "images.jpg"
if not os.path.exists(local_image_path):
    raise FileNotFoundError(f"No se encontró la imagen en: {local_image_path}")

image = Image.open(local_image_path).convert("RGB")

# 5. Construcción del payload
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe what you see in this image and what breed of animal it appears to be."}
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
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print("\n--- Qwen2.5-VL 3B Output ---")
print(output_text)

# --- NUEVO: Imprimir el tiempo de ejecución calculado ---
print("\n--- Performance Metrics ---")
print(f"Inference execution time: {inference_time:.4f} seconds")


# =============================================================================
# MODIFICACIÓN PARA CONSULTA DE SOLO TEXTO Y MEDICIÓN DE TIEMPO
# =============================================================================

# 1. Definir el prompt de texto puro
prompt = "Who is Garry Kasparov."

# 2. Estructurar el mensaje omitiendo el componente de imagen
messages = [
    {
        "role": "user",
        "content": prompt  # Al pasar el string directo, el procesador entiende que es solo texto
    }
]

# 3. Aplicar la plantilla de chat del procesador
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 4. Procesar los tensores (pasando images=None y videos=None)
inputs = processor(
    text=[text],
    images=None,
    videos=None,
    padding=True,
    return_tensors="pt"
).to(device)

print(f"\nEjecutando inferencia de solo texto en {device.upper()}...")

# 5. Medir el tiempo de ejecución de la generación de texto
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

inference_time = time.time() - start_time

# 6. Decodificar y recortar los tokens de entrada
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print("\n--- Qwen2.5-VL 3B Output (Solo Texto) ---")
print(output_text)

print("\n--- Performance Metrics ---")
print(f"Inference execution time: {inference_time:.4f} seconds")