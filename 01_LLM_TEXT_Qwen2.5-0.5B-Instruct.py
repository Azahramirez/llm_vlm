import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define a lightweight, laptop-friendly Qwen model identifier
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Automatically select hardware acceleration target
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on device: {device.upper()}")

# 3. Download and instantiate the tokenizer and causal language model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",       # Adapts precision depending on hardware capabilities
    device_map={"": device}   # Directs model orchestration to target device
)

# Extraer el directorio de almacenamiento local de forma segura ---
# Se evalúa la variable de entorno oficial o se recurre a la ruta estándar por defecto.
hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
local_path = os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")
print(f"Model and tokenizer local storage directory: {os.path.abspath(local_path)}")

# 4. Construct prompt structure using the official system/user chat schema
prompt = "Factorial de 5"
messages = [
    {"role": "system", "content": "You are a concise, technically accurate assistant."},
    {"role": "user", "content": prompt}
]

# 5. Process prompt text into system-aligned raw tensor inputs
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 6. Execute autoregressive text generation
print("\nGenerating response...")

# --- Iniciar temporizador de inferencia ---
if device == "cuda":
    torch.cuda.synchronize()  # Sincroniza operaciones asíncronas de la GPU antes de medir
start_time = time.time()

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=1.0
)

# --- Finalizar temporizador de inferencia ---
if device == "cuda":
    torch.cuda.synchronize()  # Asegura que la GPU terminó el procesamiento
inference_time = time.time() - start_time

# 7. Isolate newly generated token ids and decode to string output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Qwen Output ---")
print(response)

# --- Imprimir el tiempo de ejecución calculado ---
print("\n--- Performance Metrics ---")
print(f"Inference execution time: {inference_time:.4f} seconds")