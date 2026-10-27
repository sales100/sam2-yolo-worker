"""
Minimal SAM2 click-to-box server for the OMS annotation editor.

  GET  /health   -> {"ok": true, "device": "cuda", "model": "..."}       (Bearer key required)
  POST /segment  {image_b64, points:[[x,y],...], labels:[1,...], normalised:true}
                 -> {"box": {"x","y","w","h"}, "score": float}  (normalised 0..1, top-left + size)

Images are processed in memory only; nothing is written to disk or logged.
Env: SAM2_API_KEY (required, long random string), SAM2_MODEL (default facebook/sam2.1-hiera-small),
     SAM2_MAX_BYTES (default 25 MB).
Run: uvicorn server:app --host 127.0.0.1 --port 8765
"""
import base64
import hmac
import io
import os

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from sam2.sam2_image_predictor import SAM2ImagePredictor

API_KEY = os.environ.get("SAM2_API_KEY", "")
if len(API_KEY) < 24:
    raise SystemExit("Set SAM2_API_KEY to a random string of at least 24 characters")
MODEL_ID = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-small")
MAX_BYTES = int(os.environ.get("SAM2_MAX_BYTES", str(25 * 1024 * 1024)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=DEVICE)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def auth(authorization: str = Header(default="")):
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(token.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="unauthorised")


class SegmentRequest(BaseModel):
    image_b64: str
    points: list[list[float]] = Field(min_length=1, max_length=20)
    labels: list[int] = Field(min_length=1, max_length=20)
    normalised: bool = True


@app.get("/health", dependencies=[Depends(auth)])
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_ID}


@app.post("/segment", dependencies=[Depends(auth)])
def segment(req: SegmentRequest):
    if len(req.image_b64) > MAX_BYTES * 4 // 3 + 4:
        raise HTTPException(status_code=413, detail="image too large")
    if len(req.points) != len(req.labels):
        raise HTTPException(status_code=400, detail="points/labels length mismatch")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")
    w, h = img.size
    pts = np.array(req.points, dtype=np.float32)
    if req.normalised:
        pts = pts * np.array([w, h], dtype=np.float32)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        predictor.set_image(np.array(img))
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=np.array(req.labels), multimask_output=True)
    best = int(np.argmax(scores))
    ys, xs = np.nonzero(masks[best] > 0)
    if xs.size == 0:
        return {"box": None, "score": float(scores[best])}
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return {"box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}, "score": float(scores[best])}
