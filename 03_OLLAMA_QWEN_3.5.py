import ollama
import time

# ================== CONFIGURACIÓN ==================
#model_name = "qwen3.5:0.8b"   
model_name = "qwen3.5:9b"
#model_name = "qwen3.5:2b-q8_0"
#model_name = "qwen3.5:2b-q4_K_M"
#model_name = "qwen3.5:4b-q8_0"


# Parámetros comunes

params = {
    "temperature": 0.5,      # 0.0 a 2.0
    "top_p": 0.9,            # 0.0 a 1.0
    "top_k": 40,             # Número de tokens candidatos
    "max_tokens": 100,      # Límite de respuesta
    "repeat_penalty": 1.1,   # Penalización por repetición (1.0 = sin penalización)
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop": None,            # Palabras para detener generación
}

print(f"Modelo: {model_name}\n")

# ================== FUNCIÓN PARA CHAT ==================
def preguntar(pregunta, imagen=None):
    start_time = time.time()
    
    messages = [{'role': 'user', 'content': pregunta}]
    
    if imagen:
        messages[0]['images'] = [imagen]
    
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options=params          # Aquí se pasan los parámetros
    )
    
    end_time = time.time()
    tiempo = end_time - start_time
    
    print(f"\n⏱ Tiempo de inferencia: {tiempo:.2f} segundos")
    print(f"Respuesta:\n{response['message']['content']}\n")
    return response['message']['content']


# ================== EJEMPLOS DE USO ==================

# Pregunta sin imagen
preguntar("¿Quién es Garry Kasparov?")

# Pregunta con imagen (descomenta si quieres probar)
preguntar("Describe detalladamente esta imagen", imagen="images.jpg")#"dog.jpg")