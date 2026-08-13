
"""
Training script for a YOLO model with husky detection and also all the other 80 COCO classes.
"""

from ultralytics import YOLO

# ============================== CONFIG ==============================
BASE_WEIGHTS = "yolov8s.pt"          # COCO-pretrained checkpoint (80 classes)
DATA_YAML = "husky_dataset/dataset.yaml"  # produced by prepare_finetune_dataset.py
CERTAIN_ROWS_ONLY = True             # if True, only process the rows in selected_rows below

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8                        # lower if you hit VRAM limits
PROJECT_DIR = "runs_husky"
RUN_NAME = "yolov8s_husky"

# Freezing the backbone (and most of the neck) preserves the COCO-learned
# features and reduces catastrophic forgetting of the original 80 classes,
# since we're only training on husky-only images with no COCO annotations
# to reinforce them. Set to 0 to fine-tune the whole network instead.
FREEZE_LAYERS = 10
# ======================================================================


def main():
    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        freeze=FREEZE_LAYERS,
        patience=20,        # early stopping if val loss plateaus
        pretrained=True,
    )

    metrics = model.val(data=DATA_YAML)
    print("\n--- Validation metrics ---")
    print(metrics)


if __name__ == "__main__":
    main()