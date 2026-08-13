"""
Prepare a dataset for training a YOLO model with husky detection, organize the folders in the required structure for ultralytics.
"""

import os
import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ============================== CONFIG ==============================
PARQUET_PATH = "dataset/train-00000-of-00001.parquet"
IMAGE_COLUMN = "image"

LABEL_DIR = "yolo_labelsDIFF"        # corrected YOLO .txt files (class 0 = husky)
DATASET_ROOT = "husky_dataset2"     # output dataset root (ultralytics-style layout)

N_TRAIN = 0                     # first 70 -> train
N_VAL = 30                         # last 30 -> val

HUSKY_CLASS_ID = 80                # appended after COCO's 80 classes (indices 0-79)
CERTAIN_ROWS_ONLY = True             # if True, only process the rows in selected_rows below
# ======================================================================

# Standard 80 COCO class names, in the official index order (0-79).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
selected_rows =[
    154, 163, 171, 175, 188, 213, 216, 234, 252, 263,
    276, 310, 325, 335, 338, 344, 367, 1213, 1217, 1218,
    1249, 1270, 1277, 1287, 1296, 1298, 1112, 1115, 1123,
    1144, 1151
    ] # Example: select rows by index
assert len(COCO_CLASSES) == 80, "COCO class list must have exactly 80 entries"


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
    """'00001_husky' -> (1, 'husky')."""
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0]), parts[1]
    return None, stem


def remap_label_file(src_txt, dst_txt):
    """Copies a YOLO label file, remapping class 0 (husky) -> HUSKY_CLASS_ID."""
    lines_out = []
    with open(src_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id == 0:
                cls_id = HUSKY_CLASS_ID
            lines_out.append(" ".join([str(cls_id)] + parts[1:]))

    with open(dst_txt, "w") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")


def build_split(rows, split_name, df):
    img_dir = Path(DATASET_ROOT) / "images" / split_name
    lbl_dir = Path(DATASET_ROOT) / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for label_path in rows:
        stem = label_path.stem
        idx, label_val = parse_stem(stem)
        print(f"[{split_name}] Processing {stem} -> idx={idx}, label='{label_val}'")

        if idx is None or idx >= len(df):
            print(f"[SKIP] No matching parquet row for {stem}")
            continue

        row = df.iloc[idx]
        try:
            pil_image = load_image_from_cell(row[IMAGE_COLUMN])
        except Exception as e:
            print(f"[SKIP] Could not decode image for {stem}: {e}")
            continue

        out_img_path = img_dir / f"{stem}.jpg"
        pil_image.save(out_img_path, quality=95)

        out_lbl_path = lbl_dir / f"{stem}.txt"
        remap_label_file(label_path, out_lbl_path)

        written += 1

    print(f"  {split_name}: {written} images/labels written -> {img_dir} / {lbl_dir}")


def write_dataset_yaml():
    class_names = COCO_CLASSES + ["husky"]
    yaml_path = Path(DATASET_ROOT) / "dataset.yaml"

    lines = []
    lines.append(f"path: {os.path.abspath(DATASET_ROOT)}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndataset.yaml written -> {yaml_path} ({len(class_names)} classes)")


def main():
    df = pd.read_parquet(PARQUET_PATH)
    #if CERTAIN_ROWS_ONLY:
    #    df = df.iloc[selected_rows]
    #df = df.reset_index(drop=True)
    #print(df.head())

    # reset index to ensure sequential numbering for train/val split
   
    print(f"Dataset cargado: {len(df)} filas.")

    labels = sorted(Path(LABEL_DIR).glob("*.txt"))
    labels = [p for p in labels if p.stem != "classes"]

    if len(labels) < N_TRAIN + N_VAL:
        print(f"[WARNING] Found {len(labels)} label files, expected at least "
              f"{N_TRAIN + N_VAL} for a {N_TRAIN}/{N_VAL} split.")

    train_labels = labels[:N_TRAIN]
    val_labels = labels[N_TRAIN:N_TRAIN + N_VAL]

    print(f"\nSplitting {len(labels)} labeled images -> "
          f"{len(train_labels)} train / {len(val_labels)} val")

    if Path(DATASET_ROOT).exists():
        print(f"[INFO] {DATASET_ROOT} already exists, files will be overwritten in place.")

    build_split(train_labels, "train", df)
    build_split(val_labels, "val", df)
    write_dataset_yaml()

    print("\n✅ Dataset listo para entrenamiento.")


if __name__ == "__main__":
    main()