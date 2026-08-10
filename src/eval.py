import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================== CONFIG ==============================
WEIGHTS_PATH = "best.pt"   # fine-tuned checkpoint
DATA_YAML = "husky_dataset2/dataset.yaml"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25          # confidence threshold used for prediction visualization 0.25 originally
IOU_THRESHOLD = 0.5            # IoU threshold for NMS during prediction

PROJECT_DIR = "runs_husky"
EVAL_RUN_NAME = "eval_val"
PREDICT_RUN_NAME = "predict_val"

VAL_IMAGES_DIR = "husky_dataset2/images/val"   # used to generate visual predictions

# --- Latency / FPS benchmark config ---
N_WARMUP = 10        # inference calls discarded before timing starts (GPU/cuDNN warmup)
N_TIMED_RUNS = 50     # total timed inference calls (cycles through val images if fewer exist)

HUSKY_CLASS_ID = 80   # husky's index in dataset.yaml (appended after COCO's 80 classes)
TOP_N_CONFUSIONS = 5  # how many other classes to show in the focused husky confusion plot
# ======================================================================


def print_metrics(metrics, class_names):
    print("\n===================================")
    print("Overall metrics (all 81 classes)")
    print("===================================")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Per-class AP50 / AP50-95, isolate husky specifically
    print("\n===================================")
    print("Per-class AP (only classes present in val set)")
    print("===================================")

    ap50_per_class = metrics.box.ap50   # array aligned with metrics.box.ap_class_index
    ap_per_class = metrics.box.ap

    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = class_names[int(cls_idx)]
        marker = "  <-- husky" if cls_name == "husky" else ""
        print(f"  [{int(cls_idx):>2}] {cls_name:<15} "
              f"AP50={ap50_per_class[i]:.4f}  AP50-95={ap_per_class[i]:.4f}{marker}")


def print_husky_confusion(metrics, class_names, save_dir):
    """Extracts and prints the confusion-matrix numbers for the husky class
    (index HUSKY_CLASS_ID), and saves a focused heatmap plot isolating husky
    against the classes it's most often confused with (plus background)."""

    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is None or cm_obj.matrix is None:
        print("[WARNING] No se encontró confusion_matrix en los resultados de "
              "model.val() (asegúrate de que plots=True).")
        return

    matrix = cm_obj.matrix  # shape (nc+1, nc+1); matrix[pred, true]
    n = matrix.shape[0]
    background_idx = n - 1  # last row/col = background / no-detection

    if HUSKY_CLASS_ID >= n:
        print(f"[WARNING] HUSKY_CLASS_ID={HUSKY_CLASS_ID} está fuera de rango "
              f"para una matriz de {n}x{n}. Revisa que el checkpoint tenga "
              f"la clase husky correctamente registrada.")
        return

    tp = matrix[HUSKY_CLASS_ID, HUSKY_CLASS_ID]

    # False negatives: true husky, predicted as something else (or missed = background)
    fn_col = matrix[:, HUSKY_CLASS_ID].copy()
    fn_col[HUSKY_CLASS_ID] = 0
    fn_total = fn_col.sum()
    fn_missed = matrix[background_idx, HUSKY_CLASS_ID]   # husky present, nothing detected

    # False positives: predicted husky, but true class was something else (or background)
    fp_row = matrix[HUSKY_CLASS_ID, :].copy()
    fp_row[HUSKY_CLASS_ID] = 0
    fp_total = fp_row.sum()
    fp_background = matrix[HUSKY_CLASS_ID, background_idx]  # husky predicted on empty background

    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0

    print("\n===================================")
    print(f"Matriz de confusión - clase 'husky' (índice {HUSKY_CLASS_ID})")
    print("===================================")
    print(f"True Positives (husky correctamente detectado): {int(tp)}")
    print(f"False Negatives (husky real no detectado / mal clasificado): {int(fn_total)}")
    print(f"  de los cuales, no detectado en absoluto (fondo): {int(fn_missed)}")
    print(f"False Positives (se predijo husky y no lo era): {int(fp_total)}")
    print(f"  de los cuales, sobre fondo (sin objeto real): {int(fp_background)}")
    print(f"Precision (solo husky): {precision:.4f}")
    print(f"Recall (solo husky):    {recall:.4f}")

    # Which classes is husky most often confused with?
    confusions = []
    for idx in range(n):
        if idx == HUSKY_CLASS_ID:
            continue
        name = "background" if idx == background_idx else class_names[idx]
        # predicted husky but true was idx, plus true husky but predicted idx
        count = int(matrix[HUSKY_CLASS_ID, idx] + matrix[idx, HUSKY_CLASS_ID])
        if count > 0:
            confusions.append((name, count))

    confusions.sort(key=lambda x: x[1], reverse=True)

    if confusions:
        print("\nClases con las que más se confunde 'husky':")
        for name, count in confusions[:TOP_N_CONFUSIONS]:
            print(f"  {name:<15} {count} confusiones")
    else:
        print("\nSin confusiones registradas para 'husky' (todo TP o sin instancias).")

    print("===================================")

    # --- Focused heatmap: husky vs background + top confused classes ---
    top_names = [c[0] for c in confusions[:TOP_N_CONFUSIONS]]
    top_idx = []
    for name in top_names:
        if name == "background":
            top_idx.append(background_idx)
        else:
            top_idx.append(class_names.index(name) if isinstance(class_names, list)
                            else list(class_names.values()).index(name))

    focus_idx = [HUSKY_CLASS_ID] + top_idx
    focus_labels = ["husky"] + top_names
    sub_matrix = matrix[np.ix_(focus_idx, focus_idx)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub_matrix, cmap="Blues")
    ax.set_xticks(range(len(focus_labels)))
    ax.set_yticks(range(len(focus_labels)))
    ax.set_xticklabels(focus_labels, rotation=45, ha="right")
    ax.set_yticklabels(focus_labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion matrix - husky (focused view)")

    for i in range(sub_matrix.shape[0]):
        for j in range(sub_matrix.shape[1]):
            ax.text(j, i, int(sub_matrix[i, j]), ha="center", va="center",
                     color="black", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    save_path = "confusion_matrix_husky.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nMatriz de confusión enfocada en 'husky' guardada en: {save_path}")


def benchmark_latency_fps(model, val_images, device):
    """Measures per-image inference latency and FPS using the model's own
    forward pass (predict), excluding disk I/O from the timed region.
    Uses torch.cuda.synchronize() around each call so GPU timing is accurate
    (CUDA calls are asynchronous otherwise and would under-report latency)."""

    if not val_images:
        print(f"[WARNING] No hay imágenes en {VAL_IMAGES_DIR} para el benchmark.")
        return

    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Cycle through available images to reach N_WARMUP + N_TIMED_RUNS calls,
    # even if there are fewer than that many val images.
    def image_cycle(n):
        for i in range(n):
            yield val_images[i % len(val_images)]

    print(f"\nCalentando modelo ({N_WARMUP} inferencias, descartadas)...")
    for img_path in image_cycle(N_WARMUP):
        _ = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
            save=False,
        )
        if is_cuda:
            torch.cuda.synchronize()

    print(f"Midiendo latencia sobre {N_TIMED_RUNS} inferencias...")
    latencies_ms = []

    for img_path in image_cycle(N_TIMED_RUNS):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
            save=False,
        )

        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    latencies_ms = torch.tensor(latencies_ms)
    avg_latency_ms = latencies_ms.mean().item()
    std_latency_ms = latencies_ms.std().item()
    p95_latency_ms = latencies_ms.quantile(0.95).item()
    min_latency_ms = latencies_ms.min().item()
    max_latency_ms = latencies_ms.max().item()
    fps = 1000.0 / avg_latency_ms

    print("\n===================================")
    print(f"Latencia / FPS (device={device}, imgsz={IMG_SIZE}, "
          f"{N_TIMED_RUNS} corridas tras {N_WARMUP} de calentamiento)")
    print("===================================")
    print(f"Latencia promedio: {avg_latency_ms:.2f} ms  (+/- {std_latency_ms:.2f} ms)")
    print(f"Latencia p95:      {p95_latency_ms:.2f} ms")
    print(f"Latencia min/max:  {min_latency_ms:.2f} / {max_latency_ms:.2f} ms")
    print(f"FPS (extremo a extremo, batch=1): {fps:.2f}")
    print("===================================")

    # Ultralytics also reports its own internal timing split (preprocess /
    # inference / postprocess) during model.val() -- useful cross-check.
    print("\nNota: model.val() reporta por separado el tiempo de preprocesamiento, "
          "inferencia y postprocesamiento (ver 'Speed:' en su salida), que puedes "
          "comparar contra este benchmark end-to-end.")


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {WEIGHTS_PATH}. "
            f"Ajusta WEIGHTS_PATH a la ruta real de tu best.pt."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = YOLO(WEIGHTS_PATH)

    # 1. Quantitative evaluation on the val split (mAP50, mAP50-95, precision, recall)
    print("Ejecutando validación cuantitativa...")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=EVAL_RUN_NAME,
        conf=CONF_THRESHOLD,
        plots=True,   # saves confusion matrix, PR curve, etc. under runs_husky/eval_val
    )

    print_metrics(metrics, model.names)

    # 1b. Confusion matrix focused on the husky class specifically
    class_names_list = [model.names[i] for i in range(len(model.names))]
    print_husky_confusion(metrics, class_names_list, os.path.join(PROJECT_DIR, EVAL_RUN_NAME))

    # 2. Qualitative check: run predictions on val images and save annotated results
    print("\nGenerando predicciones visuales sobre el set de validación...")
    val_images = sorted(Path(VAL_IMAGES_DIR).glob("*.jpg"))

    if not val_images:
        print(f"[WARNING] No se encontraron imágenes en {VAL_IMAGES_DIR}")
    else:
        model.predict(
            source=VAL_IMAGES_DIR,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            project=PROJECT_DIR,
            name=PREDICT_RUN_NAME,
            save=True,
            classes=None,   # keep all classes; change to [80] to only show husky detections
        )
        print(f"Predicciones guardadas en: {os.path.join(PROJECT_DIR, PREDICT_RUN_NAME)}")

    # 3. Latency / FPS benchmark
    benchmark_latency_fps(model, val_images, device)

    print("\n===================================")
    print(f"Métricas y gráficos: {os.path.join(PROJECT_DIR, EVAL_RUN_NAME)}")
    print(f"Predicciones visuales: {os.path.join(PROJECT_DIR, PREDICT_RUN_NAME)}")
    print("===================================")


if __name__ == "__main__":
    main()