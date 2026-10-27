"""
Runpod Serverless handler: SAM2 click-to-box + YOLOv5/v8 .pt -> ONNX export (one image, scale-to-zero).

Auth is the Runpod endpoint API key (only the OMS server holds it). Images and weights are fetched from
short-lived presigned URLs, processed in memory / a temp dir that is wiped, and never logged.

input:
  {"op": "health"}
  {"op": "segment", "image_url"|"image_b64": ..., "points": [[x,y]], "labels": [1], "normalised": true}
      -> {"box": {x,y,w,h} | null, "score": float}   (normalised 0..1)
  {"op": "export", "pt_url": GET url, "put_url": PUT url, "family": "yolov5"|"yolov8", "imgsz": 640}
      -> {"bytes": int, "imgsz": int, "names": [...], "family": str, "output_shape": [...]}
"""
import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import torch
from PIL import Image

MODEL_ID = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-small")
MAX_BYTES = int(os.environ.get("SAM2_MAX_BYTES", str(25 * 1024 * 1024)))
MAX_PT_BYTES = int(os.environ.get("MAX_PT_BYTES", str(1024 * 1024 * 1024)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLOV5_DIR = os.environ.get("YOLOV5_DIR", "/opt/yolov5")
_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=DEVICE)
    return _predictor


def fetch(url: str, limit: int) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("https url required")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file too large")
    return data


def segment(inp):
    pts_in, labels = inp.get("points") or [], inp.get("labels") or []
    if not (1 <= len(pts_in) <= 20) or len(pts_in) != len(labels):
        return {"error": "points/labels"}
    raw = fetch(inp["image_url"], MAX_BYTES) if inp.get("image_url") else base64.b64decode(inp.get("image_b64", ""))
    if len(raw) > MAX_BYTES:
        return {"error": "image too large"}
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    pts = np.array(pts_in, dtype=np.float32)
    if inp.get("normalised", True):
        pts = pts * np.array([w, h], dtype=np.float32)
    p = predictor()
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        p.set_image(np.array(img))
        masks, scores, _ = p.predict(point_coords=pts, point_labels=np.array(labels), multimask_output=True)
    best = int(np.argmax(scores))
    ys, xs = np.nonzero(masks[best] > 0)
    if xs.size == 0:
        return {"box": None, "score": float(scores[best])}
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return {"box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}, "score": float(scores[best])}


def is_yolov5(pt_path: str) -> bool:
    # Original YOLOv5 checkpoints pickle classes from the yolov5 repo's "models.yolo" module.
    import zipfile
    try:
        with zipfile.ZipFile(pt_path) as z:
            blob = b"".join(z.read(n) for n in z.namelist() if n.endswith("data.pkl"))
        return b"models.yolo" in blob and b"ultralytics.nn" not in blob
    except Exception:
        return False


def export_file(pt: str, family: str, imgsz: int):
    """Export a local .pt to ONNX (opset 12). Returns (onnx_path, names, family, output_shape)."""
    family = (family or "").lower()
    if family not in ("yolov5", "yolov8"):
        family = "yolov5" if is_yolov5(pt) else "yolov8"
    names = []
    if family == "yolov5":
        r = subprocess.run(
            [sys.executable, "export.py", "--weights", pt, "--include", "onnx", "--imgsz", str(imgsz), "--opset", "12", "--simplify"],
            cwd=YOLOV5_DIR, capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            raise RuntimeError("yolov5 export failed: " + (r.stderr or r.stdout)[-800:])
        onnx = os.path.splitext(pt)[0] + ".onnx"
        try:
            sys.path.insert(0, YOLOV5_DIR)
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            n = (ck.get("model") or ck.get("ema")).names
            names = list(n.values()) if isinstance(n, dict) else list(n)
        except Exception:
            pass
    else:
        from ultralytics import YOLO
        m = YOLO(pt)
        names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
        onnx = m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False, nms=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    return onnx, names, family, list(sess.get_outputs()[0].shape)


def put_file(url: str, path: str, content_type="application/octet-stream"):
    data = open(path, "rb").read()
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=600) as r:
        if r.status not in (200, 201, 204):
            raise RuntimeError(f"upload failed {r.status}")
    return len(data)


def export(inp):
    imgsz = int(inp.get("imgsz") or 640)
    if not (160 <= imgsz <= 2048 and imgsz % 32 == 0):
        return {"error": "imgsz must be a multiple of 32 between 160 and 2048"}
    work = tempfile.mkdtemp(prefix="exp-")
    try:
        pt = os.path.join(work, "best.pt")
        with open(pt, "wb") as f:
            f.write(fetch(inp["pt_url"], MAX_PT_BYTES))
        onnx, names, family, out_shape = export_file(pt, inp.get("family") or "", imgsz)
        size = put_file(inp["put_url"], onnx)
        return {"bytes": size, "imgsz": imgsz, "names": names[:1000], "family": family, "output_shape": out_shape}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def handler(job):
    inp = job.get("input") or {}
    op = inp.get("op")
    try:
        if op == "health":
            return {"ok": True, "device": DEVICE, "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}
        if op == "segment":
            return segment(inp)
        if op == "export":
            return export(inp)
        if op == "train":  # Serverless fallback for short jobs; long jobs run as a Pod (trainer.py --pod)
            import trainer
            return trainer.run_standalone(inp)
        return {"error": "unknown op"}
    except Exception as e:  # never echo inputs (URLs are credentials)
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


if __name__ == "__main__":
    import runpod

    if os.environ.get("PRELOAD_SAM2", "1") == "1":
        predictor()
    runpod.serverless.start({"handler": handler})
