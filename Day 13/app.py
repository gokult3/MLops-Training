import os
import uuid
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH    = BASE_DIR / "best.pt"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_SIZE_MB   = 16

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE_MB * 1024 * 1024

# ── Load YOLO model once ────────────────────────────────────────────────────────
model = YOLO(str(MODEL_PATH))

# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def run_inference(img_path: Path, conf_thresh: float = 0.25) -> dict:
    """Run YOLO inference and save annotated result image."""
    t0 = time.perf_counter()
    results = model.predict(
        source=str(img_path),
        conf=conf_thresh,
        save=False,
        verbose=False,
    )
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    result      = results[0]
    orig_img    = result.orig_img          # BGR numpy array
    boxes       = result.boxes
    class_names = model.names

    # Build detection list
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        xyxy   = box.xyxy[0].tolist()      # [x1, y1, x2, y2]
        detections.append({
            "label":      class_names[cls_id],
            "class_id":   cls_id,
            "confidence": round(conf * 100, 1),
            "bbox":       [round(v, 1) for v in xyxy],
        })

    # Draw bounding boxes on image
    annotated = orig_img.copy()
    palette   = _get_palette(len(class_names))
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = palette[det["class_id"] % len(palette)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['label']}  {det['confidence']}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Save result
    result_name = f"result_{img_path.stem}.jpg"
    result_path = RESULT_FOLDER / result_name
    cv2.imwrite(str(result_path), annotated)

    # Image dimensions
    h, w = orig_img.shape[:2]

    return {
        "result_image": f"results/{result_name}",
        "detections":   detections,
        "total":        len(detections),
        "inference_ms": inference_ms,
        "image_size":   {"width": w, "height": h},
        "model":        MODEL_PATH.name,
    }


def _get_palette(n: int):
    """Generate visually distinct BGR colors."""
    import colorsys
    colors = []
    for i in range(max(n, 20)):
        h = i / max(n, 20)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXT}"}), 415

    conf_thresh = float(request.form.get("conf", 0.25))
    conf_thresh = max(0.01, min(conf_thresh, 0.99))

    # Save upload
    ext       = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    uid       = uuid.uuid4().hex[:10]
    img_name  = f"upload_{uid}.{ext}"
    img_path  = UPLOAD_FOLDER / img_name
    file.save(str(img_path))

    try:
        payload = run_inference(img_path, conf_thresh)
        payload["upload_image"] = f"uploads/{img_name}"
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR / "static", filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH.name, "classes": len(model.names)})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
