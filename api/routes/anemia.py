# api/routes/anemia.py
import os, cv2, numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as T
from PIL import Image

from ..psuedomask import run_pseudomask_on_bytes, DEFAULT_API_OUT
from api.utils.config import get_anemia_threshold

# If you kept build_model in your training script:
from training.anemia.train import build_model

router = APIRouter()

# -------------------------
# Config (via env or sane defaults)
# -------------------------
CKPT_PATH = Path(os.getenv("ANEMIA_CKPT", "runs/anemia_resnet18_v1/best_acc.pth"))
BACKBONE  = os.getenv("ANEMIA_BACKBONE", "")  # if empty, use value from ckpt
IMG_SIZE  = int(os.getenv("ANEMIA_IMG_SIZE", "224"))

# -------------------------
# Model load (once at startup)
# -------------------------
if not CKPT_PATH.is_file():
    raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

_ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
_backbone = BACKBONE or _ckpt.get("backbone", "resnet18")
_classes = _ckpt.get("classes", ["anemic", "nonanemic"])  # canonical order
_model = build_model(num_classes=2, backbone=_backbone, pretrained=False)
_model.load_state_dict(_ckpt["model"])
_model.eval()

_infer_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# -------------------------
# Helpers
# -------------------------
def _decode_bgr(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img

def _read_mask_u8(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask {path}")
    _, m2 = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m2

def _crop_with_mask(bgr: np.ndarray, mask: np.ndarray) -> Image.Image:
    # tight bbox around mask; fallback to center crop if mask is empty
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: square center crop (80% of min side)
        h, w = bgr.shape[:2]
        s = int(0.8 * min(h, w))
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = bgr[y0:y0+s, x0:x0+s]
    else:
        x0, x1 = max(xs.min()-5, 0), min(xs.max()+6, bgr.shape[1])
        y0, y1 = max(ys.min()-5, 0), min(ys.max()+6, bgr.shape[0])
        crop = bgr[y0:y1, x0:x1]
    # BGR -> RGB -> PIL
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)

def _predict_p_anemic(pil_img: Image.Image) -> float:
    x = _infer_tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # classes saved in ckpt define the index mapping
    idx = _classes.index("anemic")
    return float(probs[idx])

# -------------------------
# Routes
# -------------------------
# api/routes/anemia.py  (drop-in replacement for the route body only)
@router.post("/predict/anemia")
async def predict_anemia(file: UploadFile = File(...)):
    """
    CNN-based anemia prediction on the uploaded image *without* mask cropping.
    Returns: label, p_anemic, threshold, anemic (bool)
    """
    try:
        data = await file.read()

        # Decode full image (no mask, no crop)
        bgr = _decode_bgr(data)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # CNN inference
        p_anemic = _predict_p_anemic(pil_img)

        # Threshold (env/config), default 0.60
        threshold = get_anemia_threshold() or 0.60
        is_anemic = p_anemic >= threshold
        label = "anemic" if is_anemic else "nonanemic"

        return JSONResponse({
            "label": label,
            "p_anemic": p_anemic,
            "threshold": threshold,
            "anemic": is_anemic,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"predict/anemia failed: {e}")

