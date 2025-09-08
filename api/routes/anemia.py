# api/routes/anemia.py
import os, io, time, hashlib
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from api.utils.config import get_anemia_threshold
from training.anemia.train import build_model  # adjust import if your path differs

router = APIRouter()

# -------- Device & determinism --------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------- Config --------
CKPT_PATH = Path(os.getenv("ANEMIA_CKPT", ""))  # REQUIRED in prod (no stub)
BACKBONE_ENV = os.getenv("ANEMIA_BACKBONE", "")  # if empty, infer from ckpt
IMG_SIZE = int(os.getenv("ANEMIA_IMG_SIZE", "224"))

# -------- Globals (lazy-init / preloaded on startup) --------
_classifier = None
_xform = None
_classes = ["anemic", "nonanemic"]  # default canonical order
_backbone = BACKBONE_ENV or "resnet18"
_model_tag = "anemia"

# -------- Helpers --------
def _make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def _extract_state_dict(obj):
    """Accept {"model": sd}, {"state_dict": sd}, or raw state_dict."""
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError(
        f"Unrecognized checkpoint format ({type(obj)}; keys={list(obj.keys()) if isinstance(obj, dict) else 'n/a'})"
    )

def _load_classifier_strict():
    """
    Strict loader: requires CKPT_PATH to exist and be valid.
    Populates globals and returns (model.eval(), transform).
    Raises on any error (no stub).
    """
    global _classifier, _xform, _classes, _backbone, _model_tag

    ckpt_path = str(CKPT_PATH)
    if not ckpt_path:
        raise RuntimeError("ANEMIA_CKPT not set (required in product mode).")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Classifier checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=_device)
    # Optional metadata
    if isinstance(obj, dict):
        meta_backbone = obj.get("backbone")
        meta_classes = obj.get("classes")
        if meta_classes and isinstance(meta_classes, (list, tuple)) and "anemic" in meta_classes:
            _classes = list(meta_classes)
        _backbone = BACKBONE_ENV or meta_backbone or _backbone or "resnet18"

    # Build & load weights
    model = build_model(num_classes=2, backbone=_backbone, pretrained=False).to(_device)
    sd = _extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ISHI] load_state_dict missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[ISHI] load_state_dict unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    model.eval()

    _classifier = model
    _xform = _make_transform()
    _model_tag = os.path.basename(ckpt_path) or "anemia"
    print(f"[ISHI] Classifier ready: {_model_tag} backbone={_backbone} classes={_classes}")
    return _classifier, _xform

def preload_model():
    """Called from app startup to fail fast if model can't load."""
    return _load_classifier_strict()

def _predict_p_anemic(pil_img: Image.Image) -> float:
    if _classifier is None or _xform is None:
        raise RuntimeError("Model not ready")
    x = _xform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _classifier(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    try:
        idx_anemic = _classes.index("anemic")
    except ValueError:
        idx_anemic = 1
    return float(probs[idx_anemic])

# -------- Routes --------
@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Product-mode prediction (no stubs):
    - requires model preloaded at startup
    - field: 'image' (multipart/form-data)
    """
    if _classifier is None or _xform is None:
        # Service is up but not ready (e.g., failed to load at startup)
        raise HTTPException(status_code=503, detail="model not ready")

    # Basic input validation
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="unsupported content-type")
    raw = await image.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large")

    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")

        p_anemic = _predict_p_anemic(img)
        threshold = float(get_anemia_threshold() or 0.50)
        is_anemic = p_anemic >= threshold

        return JSONResponse({
            "anemic": bool(is_anemic),
            "score": float(p_anemic),
            "cropper": "cropper_skipped",   # update when cropper is added
            "model": _model_tag,
            "backbone": _backbone,
            "classes": _classes,
            "threshold": threshold,
            "latency_ms": int((time.time() - t0) * 1000),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")
