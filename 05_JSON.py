import ollama
import cv2
import json
import re
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "qwen3.5:9b"
IMAGE_PATH = r"C:\Users\angel\Documents\CINVESTAV\programas\3ercuatri\llm_vlm\person.jpg"  # Replace with your image path

# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception("Could not open image.")

height, width = image.shape[:2]

print(f"Original image: {width} x {height}")

# resize image to 480x270 for inference
image_resized = cv2.resize(image, (480, 270))
# save resized image to disk
cv2.imwrite("resized_image.jpg", image_resized)
image=image_resized

# ---------------------------------------------------------
# Prompt|
# ---------------------------------------------------------

prompt = f"""
You are an expert object detector.

The ORIGINAL image resolution is:

Width = {480}
Height = {270}

Return ONLY valid JSON.

The bounding boxes MUST be expressed in the ORIGINAL image coordinates.

Detect ONLY the main objects that may collide with an Autonomous Mobile Robot.

Examples:
- person
- forklift
- pallet
- table
- chair
- shelf
- box
- cart

Ignore:
- shadows
- table legs
- text
- reflections
- tiny objects

Return this exact format:

{{
  "objects":[
    {{
      "label":"forklift",
      "bbox":[xmin,ymin,xmax,ymax]
    }}
  ]
}}

Do not output markdown.
Do not explain anything.
"""
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

# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
IMAGE_PATH = r"C:\Users\angel\Documents\CINVESTAV\programas\3ercuatri\llm_vlm\resized_image.jpg"  # Replace with your image path
print(f"Using image: {IMAGE_PATH}")
print("\n--- Prompt ---")
response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [IMAGE_PATH],
        }
    ],
    options=params
)

print("waiting for response...")
text = response['message']['content']
print(f"Response: {text} ...")

# ---------------------------------------------------------
# Remove markdown if present
# ---------------------------------------------------------

text = re.sub(r"```json", "", text)
text = re.sub(r"```", "", text)

# ---------------------------------------------------------
# Parse JSON
# ---------------------------------------------------------

try:
    data = json.loads(text)
except Exception as e:

    print("Could not parse JSON")
    print(e)
    exit()

# ---------------------------------------------------------
# Draw detections
# ---------------------------------------------------------

for obj in data["objects"]:

    label = obj["label"]

    x1, y1, x2, y2 = obj["bbox"]

    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    print("-----------------------------------")
    print(label)
    print(obj["bbox"])

# ---------------------------------------------------------
# Save image
# ---------------------------------------------------------

cv2.imwrite("prediction.jpg", image)

print("\nSaved prediction.jpg")