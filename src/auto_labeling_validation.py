"""
Auto-labeling validation script for validating bounding boxes in images in YOLO format using a parquet dataset as the source of truth.
This script reads YOLO .txt label files, draws the bounding boxes on the corresponding images, and saves the annotated images to an output directory for visual inspection.
"""

import os
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# =====================================================
# Paths / Config
# =====================================================
PARQUET_PATH =  "dataset/train-00000-of-00001.parquet"    # original dataset (source of truth for images)
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

LABEL_DIR = "yolo_labelsDIFF"        # corrected YOLO .txt files
OUTPUT_DIR = "validation"          # where annotated validation images are saved

SHOW_WINDOW = True               # set True to also preview each image before saving

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image_from_cell(cell):
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
    raise ValueError(f"Formato de imagen no reconocido: {type(cell)}")


def parse_stem(stem):
    """'00001_husky' -> (1, 'husky'). Falls back to (None, stem) if it can't parse
    a leading index, so unexpected filenames don't crash the run."""
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0]), parts[1]
    return None, stem


def draw_yolo_boxes(image, label_path):
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(f"[WARNING] {label_path.name}: invalid line {line_num}")
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            # YOLO normalized -> pixel coordinates
            xc *= w
            yc *= h
            bw *= w
            bh *= h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)

            # Clamp to image boundaries
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"class {int(cls)}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return image


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Dataset cargado: {len(df)} filas.")

    labels = sorted(Path(LABEL_DIR).glob("*.txt"))
    # classes.txt isn't a per-image annotation, skip it if present
    labels = [p for p in labels if p.stem != "classes"]

    if not labels:
        print("No label files found.")
        return

    total = 0
    missing = 0

    for label_path in labels:
        stem = label_path.stem
        idx, label_val = parse_stem(stem)

        if idx is None or idx >= len(df):
            print(f"[MISSING ROW] No matching parquet row for {stem}")
            missing += 1
            continue

        row = df.iloc[idx]
        try:
            pil_image = load_image_from_cell(row[IMAGE_COLUMN])
        except Exception as e:
            print(f"[FAILED] Could not decode image for {stem}: {e}")
            continue

        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        image = draw_yolo_boxes(image, label_path)

        out_path = Path(OUTPUT_DIR) / f"{stem}.jpg"

        if SHOW_WINDOW:
            cv2.imshow("Annotated Image", image)
            cv2.waitKey(0)

        cv2.imwrite(str(out_path), image)

        total += 1
        print(f"[{total}] Validated {stem}.jpg")

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n===================================")
    print(f"Validated images : {total}")
    print(f"Missing rows     : {missing}")
    print(f"Saved to         : {os.path.abspath(OUTPUT_DIR)}")
    print("===================================")


if __name__ == "__main__":
    main()