"""
Auto-labeling correction script for fixing bounding boxes in images using a cv2-based interactive editor.
"""


import os
import io
import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ============================== CONFIG ==============================
PARQUET_PATH = "dataset/train-00000-of-00001.parquet"   # original dataset (source of truth for images)
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"
YOLO_DIR = "yolo_labels2B"           # existing YOLO .txt files to correct (in place)
CLASS_ID = 0                       # single class: husky
WINDOW_NAME = "Bbox Corrector  [drag=new box | click-inside=delete | n/p=next/prev | s=save | u=undo | q=quit]"
DISPLAY_MAX_SIDE = 1000            # downscale large images for display only (boxes still saved at full res)
# ======================================================================


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


def yolo_txt_path(idx, label_val):
    base_name = f"{idx:05d}_{str(label_val).replace(' ', '_')}"
    return os.path.join(YOLO_DIR, f"{base_name}.txt"), base_name


def read_yolo_boxes(txt_path, w_img, h_img):
    """Reads a YOLO txt file -> list of pixel-space boxes [x_min, y_min, x_max, y_max]."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = parts
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            x_center = xc * w_img
            y_center = yc * h_img
            box_w = bw * w_img
            box_h = bh * h_img
            x_min = int(round(x_center - box_w / 2))
            y_min = int(round(y_center - box_h / 2))
            x_max = int(round(x_center + box_w / 2))
            y_max = int(round(y_center + box_h / 2))
            boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def write_yolo_boxes(txt_path, boxes, w_img, h_img, class_id=CLASS_ID):
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


class ImageEditor:
    """Holds mutable editor state for the currently displayed image."""

    def __init__(self, full_img, boxes_full_res, scale):
        self.full_img = full_img              # original resolution BGR image
        self.scale = scale                     # display_size = full_size * scale
        self.boxes = [list(b) for b in boxes_full_res]   # full-res boxes, mutable
        self.history = []                      # undo stack (deep copies of self.boxes)
        self.drawing = False
        self.start_pt = None
        self.cur_pt = None
        self.dirty = False

    def push_history(self):
        self.history.append([list(b) for b in self.boxes])
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self):
        if self.history:
            self.boxes = self.history.pop()
            self.dirty = True

    def to_display(self, pt):
        return (int(pt[0] * self.scale), int(pt[1] * self.scale))

    def to_full(self, pt):
        return (int(pt[0] / self.scale), int(pt[1] / self.scale))

    def box_at_point_full(self, pt_full, margin=4):
        """Returns index of the smallest box containing pt_full, or None."""
        x, y = pt_full
        best_idx, best_area = None, None
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best_idx, best_area = i, area
        return best_idx

    def render(self):
        disp = cv2.resize(
            self.full_img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
        ) if self.scale != 1.0 else self.full_img.copy()

        for (x1, y1, x2, y2) in self.boxes:
            p1 = self.to_display((x1, y1))
            p2 = self.to_display((x2, y2))
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
            cv2.putText(disp, "husky", (p1[0], max(0, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.drawing and self.start_pt and self.cur_pt:
            cv2.rectangle(disp, self.start_pt, self.cur_pt, (0, 0, 255), 2)

        status = f"boxes: {len(self.boxes)}" + ("  *unsaved*" if self.dirty else "")
        cv2.putText(disp, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return disp


def mouse_callback(event, x, y, flags, editor: ImageEditor):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click landed inside an existing box -> delete it
        pt_full = editor.to_full((x, y))
        idx = editor.box_at_point_full(pt_full)
        if idx is not None:
            editor.push_history()
            del editor.boxes[idx]
            editor.dirty = True
        else:
            editor.drawing = True
            editor.start_pt = (x, y)
            editor.cur_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if editor.drawing:
            editor.cur_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if editor.drawing:
            editor.drawing = False
            x1_full, y1_full = editor.to_full(editor.start_pt)
            x2_full, y2_full = editor.to_full((x, y))
            x1_full, x2_full = sorted([x1_full, x2_full])
            y1_full, y2_full = sorted([y1_full, y2_full])
            if x2_full - x1_full > 3 and y2_full - y1_full > 3:
                editor.push_history()
                editor.boxes.append([x1_full, y1_full, x2_full, y2_full])
                editor.dirty = True
            editor.start_pt = None
            editor.cur_pt = None


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Dataset cargado: {len(df)} filas.")
    print("Controles: arrastrar = nueva caja | click dentro de una caja = eliminarla")
    print("           n/p = siguiente/anterior | s = guardar | u = deshacer | q = salir")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    idx = 0
    n_rows = len(df)

    while 0 <= idx < n_rows:
        row = df.iloc[idx]
        try:
            pil_image = load_image_from_cell(row[IMAGE_COLUMN])
        except Exception as e:
            print(f"[{idx}] No se pudo decodificar la imagen: {e}")
            idx += 1
            continue

        label_val = row[LABEL_COLUMN] if LABEL_COLUMN in df.columns else "husky"
        txt_path, base_name = yolo_txt_path(idx, label_val)

        full_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h_img, w_img = full_img.shape[:2]
        boxes = read_yolo_boxes(txt_path, w_img, h_img)

        scale = min(1.0, DISPLAY_MAX_SIDE / max(w_img, h_img))
        editor = ImageEditor(full_img, boxes, scale)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, editor)

        print(f"\n[{idx + 1}/{n_rows}] {base_name}  ({len(boxes)} cajas cargadas)")

        action = None
        while True:
            disp = editor.render()
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                action = "quit"
                break
            elif key == ord('n'):
                action = "next"
                break
            elif key == ord('p'):
                action = "prev"
                break
            elif key == ord('u'):
                editor.undo()
            elif key == ord('s'):
                write_yolo_boxes(txt_path, editor.boxes, w_img, h_img)
                editor.dirty = False
                print(f"  Guardado: {txt_path} ({len(editor.boxes)} cajas)")

        # Auto-save on navigation/quit if there are unsaved changes
        if editor.dirty:
            write_yolo_boxes(txt_path, editor.boxes, w_img, h_img)
            print(f"  Guardado automáticamente: {txt_path} ({len(editor.boxes)} cajas)")

        if action == "quit":
            break
        elif action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print("\n✅ Corrección finalizada.")


if __name__ == "__main__":
    main()