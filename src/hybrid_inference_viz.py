"""
Visualize the results of a hybrid pipeline of YOLOv8 and a Visual Language Model (VLM) for husky detection in images.
"""

import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

# ============================== CONFIG ==============================
YOLO_WEIGHTS_PATH = "best.pt"
VAL_IMAGES_DIR = "husky_dataset2/images/val"
VAL_LABELS_DIR = "husky_dataset2/labels/val"   # ground truth, optional but drawn if present

QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_MODEL_NAME = QWEN_MODEL_ID.split("/")[-1]

HUSKY_CLASS_ID = 80
IMG_SIZE = 640

YOLO_CANDIDATE_CONF = 0.1
YOLO_IOU_NMS = 0.5
CROP_PADDING_FRAC = 0.10

OUTPUT_DIR = f"cascade_viz_{QWEN_MODEL_NAME}"
DRAW_GT = True             # draw ground-truth boxes too, for visual comparison
QWEN_MAX_NEW_TOKENS = 5

# Colors (BGR, for cv2)
COLOR_ACCEPTED = (0, 255, 0)     # green: YOLO detected + Qwen said "yes"
COLOR_REJECTED = (0, 0, 255)     # red: YOLO detected + Qwen said "no" (discarded)
COLOR_GT = (255, 255, 0)         # cyan, dashed: ground truth
# ======================================================================

VLM_PROMPT = "Is this a husky dog inside this image crop? Answer only Yes or No."

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_qwen():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando {QWEN_MODEL_ID} en {device.upper()}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor, device


def load_gt_boxes(label_path, w_img, h_img):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, bw, bh = parts
            if int(float(cls_id)) != HUSKY_CLASS_ID:
                continue
            xc, yc, bw, bh = float(xc) * w_img, float(yc) * h_img, float(bw) * w_img, float(bh) * h_img
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            boxes.append([x1, y1, x2, y2])
    return boxes


def pad_box(box, w_img, h_img, frac):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * frac
    y1 -= bh * frac
    x2 += bw * frac
    y2 += bh * frac
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w_img - 1, int(x2))
    y2 = min(h_img - 1, int(y2))
    return x1, y1, x2, y2


def ask_qwen_is_husky(qwen_model, processor, device, crop_img):
    """Returns (is_husky: bool, raw_text: str, elapsed_seconds: float)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_img},
                {"type": "text", "text": VLM_PROMPT},
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

    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False
        )

    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    match = re.search(r"\b(yes|no)\b", output_text, re.IGNORECASE)
    if match is None:
        return False, output_text, elapsed
    return match.group(1).lower() == "yes", output_text, elapsed


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=8):
    """cv2 has no native dashed rectangle, so draw dashed lines on each edge."""
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
        length = np.linalg.norm(p2 - p1)
        if length == 0:
            return
        n_dashes = max(1, int(length // (dash_len * 2)))
        for i in range(n_dashes):
            start = p1 + (p2 - p1) * (i * 2 * dash_len) / length
            end = p1 + (p2 - p1) * min((i * 2 + 1) * dash_len, length) / length
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def main():
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        raise FileNotFoundError(f"No se encontró {YOLO_WEIGHTS_PATH}")

    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    qwen_model, qwen_processor, qwen_device = load_qwen()

    image_paths = sorted(Path(VAL_IMAGES_DIR).glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {VAL_IMAGES_DIR}")

    print(f"\n{len(image_paths)} imágenes encontradas. Generando visualización en: {OUTPUT_DIR}\n")

    n_accepted_total = 0
    n_rejected_total = 0

    for img_idx, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB")
        w_img, h_img = pil_img.size
        vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # --- Ground truth (optional, dashed cyan) ---
        if DRAW_GT:
            label_path = Path(VAL_LABELS_DIR) / f"{img_path.stem}.txt"
            gt_boxes = load_gt_boxes(label_path, w_img, h_img)
            for (gx1, gy1, gx2, gy2) in gt_boxes:
                draw_dashed_rect(vis_img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), COLOR_GT)
            if gt_boxes:
                cv2.putText(vis_img, "GT", (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, COLOR_GT, 2)

        # --- YOLO candidates ---
        results = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=YOLO_CANDIDATE_CONF,
                                      iou=YOLO_IOU_NMS, verbose=False, save=False)

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        husky_mask = clss == HUSKY_CLASS_ID
        cand_boxes = boxes_xyxy[husky_mask]
        cand_confs = confs[husky_mask]

        n_accepted = 0
        n_rejected = 0

        for box, conf in zip(cand_boxes, cand_confs):
            x1, y1, x2, y2 = [int(v) for v in box]

            px1, py1, px2, py2 = pad_box(box, w_img, h_img, CROP_PADDING_FRAC)
            crop = pil_img.crop((px1, py1, px2, py2)) if px2 > px1 and py2 > py1 else None

            if crop is None:
                is_husky, raw_text = False, "(crop inválido)"
            else:
                is_husky, raw_text, _ = ask_qwen_is_husky(qwen_model, qwen_processor, qwen_device, crop)

            color = COLOR_ACCEPTED if is_husky else COLOR_REJECTED
            verdict = "Yes" if is_husky else "No"

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label_text = f"yolo={conf:.2f} qwen={verdict}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis_img, label_text, (x1 + 2, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if is_husky:
                n_accepted += 1
            else:
                n_rejected += 1

        # Legend
        legend_y = 25
        cv2.putText(vis_img, "verde = aceptado por Qwen", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCEPTED, 2)
        cv2.putText(vis_img, "rojo = rechazado por Qwen", (10, legend_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_REJECTED, 2)
        if DRAW_GT:
            cv2.putText(vis_img, "cian punteado = ground truth", (10, legend_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT, 2)

        out_path = Path(OUTPUT_DIR) / img_path.name
        cv2.imwrite(str(out_path), vis_img)

        n_accepted_total += n_accepted
        n_rejected_total += n_rejected

        print(f"[{img_idx + 1}/{len(image_paths)}] {img_path.name}: "
              f"{n_accepted} aceptadas, {n_rejected} rechazadas -> {out_path.name}")

    print("\n===================================")
    print(f"Total aceptadas (verde): {n_accepted_total}")
    print(f"Total rechazadas (rojo): {n_rejected_total}")
    print(f"Imágenes guardadas en: {os.path.abspath(OUTPUT_DIR)}")
    print("===================================")


if __name__ == "__main__":
    main()