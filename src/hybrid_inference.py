import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

# ============================== CONFIG ==============================
YOLO_WEIGHTS_PATH = "best.pt"                       # fine-tuned YOLOv8 checkpoint
VAL_IMAGES_DIR = "husky_dataset2/images/val"
VAL_LABELS_DIR = "husky_dataset2/labels/val"         # YOLO-format ground truth (class 80 = husky)

QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
QWEN_MODEL_NAME=QWEN_MODEL_ID.split("/")[-1]  # "Qwen3.5-2B"

HUSKY_CLASS_ID = 80
IMG_SIZE = 640

# Confidence used to gather YOLO candidate boxes BEFORE the VLM filter.
# Kept low so the VLM gets a chance to reject/accept borderline detections
# and so the resulting precision-recall curve covers enough of the
# confidence range for a meaningful mAP@50. Raise it if this is too slow.
YOLO_CANDIDATE_CONF = 0.1
YOLO_IOU_NMS = 0.5

IOU_MATCH_THRESHOLD = 0.5    # IoU threshold for both mAP@50 and the confusion matrix
CROP_PADDING_FRAC = 0.10     # extra context around each box before cropping for the VLM

PROJECT_DIR = "runs_husky"
OUTPUT_RUN_NAME = "eval_yolo_qwen_pipeline"

N_WARMUP = 5           # warmup calls (both YOLO and Qwen) before timing starts
QWEN_MAX_NEW_TOKENS = 5
# ======================================================================

VLM_PROMPT = "Is this a husky dog inside this image crop? Answer only Yes or No."


# ============================== MODELS ==============================
def load_yolo():
    return YOLO(YOLO_WEIGHTS_PATH)


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
# ======================================================================


def load_gt_boxes(label_path, w_img, h_img):
    """YOLO txt -> list of [x1, y1, x2, y2] pixel boxes for the husky class."""
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


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


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
    """Returns (is_husky: bool, elapsed_seconds: float)."""
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
        print(f"  [WARNING] Respuesta VLM ambigua: '{output_text}' -> se descarta la detección")
        return False, elapsed

    return match.group(1).lower() == "yes", elapsed


def compute_ap(tp, fp, n_gt):
    """Standard all-points-interpolation AP (VOC-style), equivalent to AP@0.5
    for a single class -- with one class this value IS mAP@50.
    Returns (ap, recalls, precisions, interpolated_precisions) so the raw and
    interpolated PR curves can be plotted."""
    if n_gt == 0:
        return 0.0, np.array([]), np.array([]), np.array([])

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap), recalls, precisions, mpre[1:-1]


def plot_pr_curve(recalls, precisions, interp_precisions, ap, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    #ax.plot(recalls, precisions, marker=".", linewidth=1, alpha=0.5,
    #        label="Precision-Recall (raw)", color="steelblue")
    ax.plot(recalls, interp_precisions, linewidth=2,
            label="Interpolated (used for AP)", color="darkorange")
    ax.fill_between(recalls, 0, interp_precisions, alpha=0.15, color="darkorange")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"PR Curve  (mAP@50 = {ap:.4f}),{QWEN_MODEL_NAME}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"PR curve guardada en: {save_path}")


def plot_confidence_curves(flat_sorted, tp, fp, n_gt, save_path):
    """Precision and Recall as a function of the confidence threshold used to
    cut the sorted detection list -- helps pick a good operating conf."""
    confs = np.array([d["conf"] for d in flat_sorted])
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recalls = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(confs, precisions, label="Precision", color="seagreen")
    ax.plot(confs, recalls, label="Recall", color="crimson")
    ax.plot(confs, f1, label="F1", color="steelblue", linestyle="--")

    if len(f1) > 0:
        best_idx = int(np.argmax(f1))
        ax.axvline(confs[best_idx], color="gray", linestyle=":", alpha=0.7)
        ax.text(confs[best_idx], 0.02, f" mejor F1 @ conf={confs[best_idx]:.2f}",
                rotation=90, va="bottom", fontsize=8)

    ax.set_xlabel("Confidence threshold (score original de YOLO)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / Recall / F1 vs. Confidence - husky")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Curva confianza vs precision/recall guardada en: {save_path}")


def plot_confusion_matrix(tp, fp, fn, save_path):
    """Simple TP/FP/FN heatmap for the single husky class (post Qwen filter)."""
    matrix = np.array([[tp, fn],
                        [fp, 0]])   # rows: predicted husky/background, cols: true husky/background
    labels = ["husky", "background"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Confusion Matrix - husky({QWEN_MODEL_NAME})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                     color="black", fontsize=12)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {save_path}")


def main():
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        raise FileNotFoundError(f"No se encontró {YOLO_WEIGHTS_PATH}")

    os.makedirs(os.path.join(PROJECT_DIR, OUTPUT_RUN_NAME), exist_ok=True)

    yolo_model = load_yolo()
    qwen_model, qwen_processor, qwen_device = load_qwen()

    image_paths = sorted(Path(VAL_IMAGES_DIR).glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {VAL_IMAGES_DIR}")

    print(f"\n{len(image_paths)} imágenes de validación encontradas.")

    # --- Warmup (both models) ---
    print(f"\nCalentando modelos ({N_WARMUP} imágenes descartadas)...")
    for img_path in image_paths[:min(N_WARMUP, len(image_paths))]:
        _ = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=YOLO_CANDIDATE_CONF,
                                iou=YOLO_IOU_NMS, verbose=False, save=False)
        pil_img = Image.open(img_path).convert("RGB")
        _ = ask_qwen_is_husky(qwen_model, qwen_processor, qwen_device, pil_img)
    if qwen_device.startswith("cuda"):
        torch.cuda.synchronize()

    # --- Main pipeline over all val images ---
    all_detections = []   # each: {"conf": float, "tp": 0/1 filled later, "img_idx": i}
    total_gt = 0

    yolo_latencies_ms = []
    vlm_latencies_ms = []       # every individual crop check
    per_image_total_ms = []     # yolo + all crop VLM calls for that image

    per_image_results = []      # for confusion-matrix bookkeeping after AP sorting

    print("\nProcesando pipeline YOLO -> Qwen por imagen...\n")
    for img_idx, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB")
        w_img, h_img = pil_img.size

        label_path = Path(VAL_LABELS_DIR) / f"{img_path.stem}.txt"
        gt_boxes = load_gt_boxes(label_path, w_img, h_img)
        total_gt += len(gt_boxes)

        # --- YOLO detection (timed) ---
        if qwen_device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=YOLO_CANDIDATE_CONF,
                                      iou=YOLO_IOU_NMS, verbose=False, save=False)
        if qwen_device.startswith("cuda"):
            torch.cuda.synchronize()
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        yolo_latencies_ms.append(yolo_ms)

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        husky_mask = clss == HUSKY_CLASS_ID
        cand_boxes = boxes_xyxy[husky_mask]
        cand_confs = confs[husky_mask]

        # --- Qwen filter per candidate crop (timed individually) ---
        kept_boxes, kept_confs = [], []
        image_vlm_ms_total = 0.0

        for box, conf in zip(cand_boxes, cand_confs):
            px1, py1, px2, py2 = pad_box(box, w_img, h_img, CROP_PADDING_FRAC)
            if px2 <= px1 or py2 <= py1:
                continue
            crop = pil_img.crop((px1, py1, px2, py2))

            is_husky, vlm_elapsed = ask_qwen_is_husky(qwen_model, qwen_processor, qwen_device, crop)
            vlm_latencies_ms.append(vlm_elapsed * 1000.0)
            image_vlm_ms_total += vlm_elapsed * 1000.0

            if is_husky:
                kept_boxes.append(box)
                kept_confs.append(conf)

        per_image_total_ms.append(yolo_ms + image_vlm_ms_total)

        per_image_results.append({
            "gt_boxes": gt_boxes,
            "det_boxes": kept_boxes,
            "det_confs": kept_confs,
        })

        n_before = int(husky_mask.sum())
        n_after = len(kept_boxes)
        print(f"[{img_idx + 1}/{len(image_paths)}] {img_path.name}: "
              f"{n_before} candidatos YOLO -> {n_after} tras filtro Qwen "
              f"(YOLO {yolo_ms:.1f}ms, VLM total {image_vlm_ms_total:.1f}ms)")

    # ============================== mAP@50 ==============================
    # Flatten all kept detections across the whole val set, sorted by
    # confidence descending, then greedily match to GT boxes per image.
    flat = []
    for img_idx, res in enumerate(per_image_results):
        for box, conf in zip(res["det_boxes"], res["det_confs"]):
            flat.append({"img_idx": img_idx, "box": box, "conf": conf})
    flat.sort(key=lambda d: d["conf"], reverse=True)

    matched_gt = [set() for _ in per_image_results]  # indices of GT already matched, per image
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))

    for i, det in enumerate(flat):
        img_idx = det["img_idx"]
        gt_boxes = per_image_results[img_idx]["gt_boxes"]

        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt[img_idx]:
                continue
            iou_val = iou(det["box"], gt_box)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j

        if best_iou >= IOU_MATCH_THRESHOLD:
            tp[i] = 1
            matched_gt[img_idx].add(best_j)
        else:
            fp[i] = 1

    map50, curve_recalls, curve_precisions, curve_interp_precisions = compute_ap(tp, fp, total_gt)

    # ============================== Confusion matrix (final set) ==============================
    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    total_fn = total_gt - total_tp   # GT boxes never matched by any kept detection

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    # ============================== Latency / FPS ==============================
    yolo_latencies_ms = np.array(yolo_latencies_ms)
    vlm_latencies_ms = np.array(vlm_latencies_ms) if vlm_latencies_ms else np.array([0.0])
    per_image_total_ms = np.array(per_image_total_ms)

    avg_pipeline_ms = per_image_total_ms.mean()
    fps_pipeline = 1000.0 / avg_pipeline_ms if avg_pipeline_ms > 0 else 0.0

    # ============================== Report ==============================
    print("\n===================================")
    print("mAP@50 (pipeline YOLO + filtro Qwen)")
    print("===================================")
    print(f"mAP@50 (husky, único class): {map50:.4f}")
    print(f"Total GT huskies: {total_gt}")
    print(f"Total detecciones finales (tras filtro Qwen): {len(flat)}")

    print("\n===================================")
    print("Matriz de confusión - clase 'husky' (tras filtro Qwen)")
    print("===================================")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\n===================================")
    print("Latencia / FPS")
    print("===================================")
    print(f"YOLO solo   -> promedio: {yolo_latencies_ms.mean():.2f} ms  "
          f"(min {yolo_latencies_ms.min():.2f} / max {yolo_latencies_ms.max():.2f})")
    print(f"Qwen (por crop) -> promedio: {vlm_latencies_ms.mean():.2f} ms  "
          f"(min {vlm_latencies_ms.min():.2f} / max {vlm_latencies_ms.max():.2f}, "
          f"n={len(vlm_latencies_ms)} crops evaluados)")
    print(f"Pipeline completo (YOLO + todos los crops Qwen) por imagen -> "
          f"promedio: {avg_pipeline_ms:.2f} ms")
    print(f"FPS del pipeline completo: {fps_pipeline:.2f}")
    print("===================================")

    # ============================== Gráficas ==============================
    run_dir = os.path.join(PROJECT_DIR, OUTPUT_RUN_NAME)

    if len(flat) > 0:
        plot_pr_curve(
            curve_recalls, curve_precisions, curve_interp_precisions, map50,
            os.path.join(run_dir, f"pr_curve_husky{QWEN_MODEL_NAME}.png"),
        )
        plot_confidence_curves(
            flat, tp, fp, total_gt,
            os.path.join(run_dir, f"confidence_curve_husky{QWEN_MODEL_NAME}.png"),
        )
    else:
        print("[WARNING] No hay detecciones finales; se omiten PR curve y "
              "curva de confianza.")

    plot_confusion_matrix(
        total_tp, total_fp, total_fn,
        os.path.join(run_dir, f"confusion_matrix_husky{QWEN_MODEL_NAME}.png"),
    )


if __name__ == "__main__":
    main()