import json
import os
import re
import time
import io

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

# ============================== CONFIG ==============================
PARQUET_PATH = "dataset/train-00000-of-00001.parquet"      # path to your parquet file
IMAGE_COLUMN = "image"                # column holding the image
LABEL_COLUMN = "label"                # class label column (used only for output filenames)
OUTPUT_DIR = "bbox_results4B"           # folder for annotated images
YOLO_DIR = "yolo_labels4B"              # folder for YOLO-format .txt files
MODEL_ID = "Qwen/Qwen3.5-2B"          # model to use for VLM inference
MAX_NEW_TOKENS = 1024
LIMIT = None                          # set to an int to only process first N rows while testing

CLASS_NAMES = ["husky"]               # single class -> class_id 0
# ======================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando modelo {MODEL_ID} en el dispositivo: {device.upper()}")

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": device},
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Single generic label per detection ("husky"), no numbering (husky1, husky2, ...)
PROMPT_TEXT = """Detect all distinct huskies in the image.

Return ONLY a valid JSON array, nothing else (no markdown fences, no explanation).
Each element must be an object with exactly these two keys:
- "label": always the string "husky" (do not number or index it)
- "bbox_2d": [x1, y1, x2, y2] normalized to a 0-1000 scale relative to the image
  width and height, where (x1, y1) is the top-left corner and (x2, y2) is the
  bottom-right corner.

Include one array element per husky detected. If you are unsure about an object,
omit it rather than guessing.

Example of the exact format required for an image with two huskies:
[{"label": "husky", "bbox_2d": [120, 340, 610, 900]}, {"label": "husky", "bbox_2d": [50, 60, 300, 400]}]
"""


def load_image_from_cell(cell):
    """Handles the common parquet image encodings: raw bytes, {'bytes': ...} dict,
    a file path string, or an already-decoded PIL/np image."""
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")

    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")

    if isinstance(cell, str) and os.path.exists(cell):
        return Image.open(cell).convert("RGB")

    if isinstance(cell, np.ndarray):
        return Image.fromarray(cell).convert("RGB")

    raise ValueError(f"Formato de imagen no reconocido en la celda: {type(cell)}")


def parse_vlm_string(vlm_string):
    """Extrae y repara el array JSON de la respuesta del modelo."""
    cleaned = re.sub(r"^```json\s*", "", vlm_string.strip())
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("[")
    if start == -1:
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


def run_inference(pil_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def clip(value, lo, hi):
    return max(lo, min(value, hi))


def detections_to_pixel_boxes(detections, w_img, h_img):
    """Converts raw 0-1000 normalized bbox_2d detections into clipped pixel-space
    boxes, all under the single 'husky' class."""
    boxes = []
    for obj in detections:
        bbox = obj.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        x_min = int(x1 / 1000 * w_img)
        y_min = int(y1 / 1000 * h_img)
        x_max = int(x2 / 1000 * w_img)
        y_max = int(y2 / 1000 * h_img)

        x_min, x_max = sorted([x_min, x_max])
        y_min, y_max = sorted([y_min, y_max])
        x_min = clip(x_min, 0, w_img - 1)
        x_max = clip(x_max, 0, w_img - 1)
        y_min = clip(y_min, 0, h_img - 1)
        y_max = clip(y_max, 0, h_img - 1)

        if x_max <= x_min or y_max <= y_min:
            continue

        boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def draw_boxes(pil_image, boxes):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    color = (0, 255, 0)

    for (x_min, y_min, x_max, y_max) in boxes:
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, "husky", (x_min, max(0, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def write_yolo_txt(txt_path, boxes, w_img, h_img, class_id=0):
    """YOLO format: one line per box -> class_id x_center y_center width height
    (all normalized 0-1 relative to image width/height)."""
    lines = []
    for (x_min, y_min, x_max, y_max) in boxes:
        box_w = x_max - x_min
        box_h = y_max - y_min
        x_center = x_min + box_w / 2
        y_center = y_min + box_h / 2

        lines.append(
            f"{class_id} "
            f"{x_center / w_img:.6f} "
            f"{y_center / h_img:.6f} "
            f"{box_w / w_img:.6f} "
            f"{box_h / h_img:.6f}"
        )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Dataset cargado: {len(df)} filas. Columnas: {list(df.columns)}")

    n_rows = len(df) if LIMIT is None else min(LIMIT, len(df))

    # classes.txt for reference (YOLO convention: line N = class_id N)
    with open(os.path.join(YOLO_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    for idx in range(n_rows):
        row = df.iloc[idx]
        try:
            pil_image = load_image_from_cell(row[IMAGE_COLUMN])
        except Exception as e:
            print(f"[{idx}] No se pudo decodificar la imagen: {e}")
            continue

        label_val = row[LABEL_COLUMN] if LABEL_COLUMN in df.columns else "husky"
        w_img, h_img = pil_image.size

        start = time.time()
        output_text = run_inference(pil_image)
        elapsed = time.time() - start

        detections = parse_vlm_string(output_text)
        boxes = detections_to_pixel_boxes(detections, w_img, h_img)
        annotated_img = draw_boxes(pil_image, boxes)

        base_name = f"{idx:05d}_{str(label_val).replace(' ', '_')}"

        img_out_path = os.path.join(OUTPUT_DIR, f"{base_name}.jpg")
        cv2.imwrite(img_out_path, annotated_img)

        txt_out_path = os.path.join(YOLO_DIR, f"{base_name}.txt")
        write_yolo_txt(txt_out_path, boxes, w_img, h_img, class_id=0)

        print(f"[{idx+1}/{n_rows}] {base_name} -> {len(boxes)} cajas "
              f"({elapsed:.2f}s)")

    print(f"\n✅ Listo.")
    print(f"Imágenes anotadas: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Etiquetas YOLO:    {os.path.abspath(YOLO_DIR)}")


if __name__ == "__main__":
    main()