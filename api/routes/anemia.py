# api/routes/anemia.py
import io
import os
import time
import hashlib
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response

from api.models.loader import load_model_from_ckpt
from api.utils.config import get_anemia_threshold

# Try to import pseudomask helper (optional; tolerate different paths/typos)
run_pseudomask_on_bytes = None
for _cand in (
    "api.utils.pseudomask",
    "api.utils.psuedomask",
    "api.psuedomask",
    "api.pseudomask",
    "..utils.pseudomask",
    "..psuedomask",
):
    try:
        mod = __import__(_cand, fromlist=["run_pseudomask_on_bytes"])
        run_pseudomask_on_bytes = getattr(mod, "run_pseudomask_on_bytes", None)
        if run_pseudomask_on_bytes:
            break
    except Exception:
        pass

router = APIRouter()

# -------- Device & determinism --------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------- Config --------
CKPT_PATH = Path(os.getenv("ANEMIA_CKPT", ""))  # REQUIRED at runtime
IMG_SIZE = int(os.getenv("ANEMIA_IMG_SIZE", "224"))
MAX_IMAGE_BYTES = int(os.getenv("ANEMIA_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))  # 5MB
ANEMIA_ARCH = os.getenv("ANEMIA_ARCH", "auto")  # "auto" | "resnet18" | "simple_cnn"

# -------- Globals --------
_classifier = None
_xform = None
_classes = ["anemic", "nonanemic"]  # canonical order
_backbone = "unknown"
_model_tag = "anemia"

# -------- Transforms --------
def _make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        # If your training used ImageNet normalization, keep this:
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -------- Checkpoint helpers (arch inference for ANEMIA_ARCH=auto) --------
def _extract_state_dict(obj):
    """Accept {'model': sd}, {'state_dict': sd}, or raw state_dict."""
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

def _infer_arch_from_keys(sd_keys):
    ks = list(sd_keys)
    if any(k.startswith(("conv1.", "bn1.", "layer1.")) for k in ks):
        return "resnet18"
    if any(k.startswith("cnn.") for k in ks):
        return "simple_cnn"
    # Default to resnet18 if unclear
    return "resnet18"

def _resolve_arch_from_ckpt(arch_env: str, ckpt_path: Path) -> str:
    if arch_env != "auto":
        return arch_env
    obj = torch.load(str(ckpt_path), map_location="cpu")
    sd = _extract_state_dict(obj)
    return _infer_arch_from_keys(sd.keys())

# -------- Loader (runtime-only) --------
def _load_classifier_strict():
    """Loads model via runtime loader (no training imports)."""
    global _classifier, _xform, _classes, _backbone, _model_tag

    ckpt_path = str(CKPT_PATH)
    if not ckpt_path:
        raise RuntimeError("ANEMIA_CKPT not set.")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Classifier checkpoint not found: {ckpt_path}")

    arch = _resolve_arch_from_ckpt(ANEMIA_ARCH, CKPT_PATH)
    model = load_model_from_ckpt(arch, ckpt_path=str(CKPT_PATH))
    model.to(_device).eval()

    # Optional: class names stored in the checkpoint dict
    try:
        raw = torch.load(str(CKPT_PATH), map_location="cpu")
        if isinstance(raw, dict):
            meta_classes = raw.get("classes")
            if meta_classes and isinstance(meta_classes, (list, tuple)) and "anemic" in meta_classes:
                _classes = list(meta_classes)
    except Exception:
        pass

    _classifier = model
    _xform = _make_transform()
    _backbone = arch
    _model_tag = os.path.basename(ckpt_path) or arch
    print(f"[ISHI] Classifier ready: tag={_model_tag} arch={_backbone} classes={_classes}")
    return _classifier, _xform

def preload_model():
    """Optionally call this from app startup to fail fast if model can't load."""
    return _load_classifier_strict()

def _get_model():
    global _classifier, _xform
    if _classifier is None or _xform is None:
        _load_classifier_strict()
    return _classifier, _xform

# -------- Inference --------
def _predict_p_anemic(pil_img: Image.Image) -> float:
    model, xform = _get_model()
    x = xform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    try:
        idx_anemic = _classes.index("anemic")
    except ValueError:
        idx_anemic = 1
    return float(probs[idx_anemic])

def _env_true(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "on")

ISHI_DISABLE_CROPPER = _env_true(os.getenv("ISHI_DISABLE_CROPPER", "0"))

# -------- Crop helpers --------
def _try_mask_crop(raw: bytes) -> Tuple[Optional[Image.Image], str]:
    """Try to crop conjunctiva using a mask; return (PIL image or None, tag)."""
    if ISHI_DISABLE_CROPPER or run_pseudomask_on_bytes is None:
        return None, "cropper_skipped"
    try:
        out = run_pseudomask_on_bytes(raw)  # may be dict/bytes/np.ndarray/str

        # Decode original as BGR (for cropping)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "cropper_decode_fail"

        mask = None

        # 1) dict style
        if isinstance(out, dict):
            mask = out.get("mask")
            if mask is None:
                mask_path = out.get("mask_path")
                if mask_path:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2) raw bytes style (encoded image)
        elif isinstance(out, (bytes, bytearray, memoryview)):
            marr = np.frombuffer(out, dtype=np.uint8)
            mask = cv2.imdecode(marr, cv2.IMREAD_GRAYSCALE)

        # 3) numpy array style
        elif isinstance(out, np.ndarray):
            mask = out
            if mask.ndim == 3:  # RGB/BGR -> gray
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 4) path string style
        elif isinstance(out, str):
            mask = cv2.imread(out, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return None, "cropper_no_mask"

        # Binarize & bbox
        _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(binm > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, "cropper_empty"

        pad = 5
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, bgr.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, bgr.shape[0])
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None, "cropper_bbox_empty"

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), "mask_crop"

    except Exception as e:
        print("[ISHI] cropper error:", type(e).__name__, str(e))
        print(traceback.format_exc())
        return None, "cropper_error"

def _center_square_crop(raw: bytes, frac: float = 0.8):
    """Heuristic fallback: center square crop."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s = int(min(h, w) * frac)
    y0 = max((h - s) // 2, 0)
    x0 = max((w - s) // 2, 0)
    crop = bgr[y0:y0 + s, x0:x0 + s]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# -------- Routes --------
@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Prediction endpoint:
      - field: 'image' (multipart/form-data)
      - arch: ANEMIA_ARCH env; 'auto' infers from ckpt keys
    """
    # Basic input validation
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="unsupported content-type")
    raw = await image.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")

    t0 = time.time()
    try:
        # Try cropper → center crop → EXIF-corrected full image
        cropped, crop_tag = _try_mask_crop(raw)
        if cropped is None:
            cc = _center_square_crop(raw, frac=0.8)
            if cc is not None:
                img = cc
                crop_tag = "center_crop"
            else:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                crop_tag = "cropper_skipped"
        else:
            img = cropped

        p_anemic = _predict_p_anemic(img)
        threshold = float(get_anemia_threshold() or 0.50)
        is_anemic = p_anemic >= threshold

        return JSONResponse({
            "anemic": bool(is_anemic),
            "score": float(p_anemic),
            "cropper": crop_tag,
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

@router.post("/debug/crop")
async def debug_crop(image: UploadFile = File(...)):
    raw = await image.read()
    cropped, tag = _try_mask_crop(raw)
    if cropped is None:
        cc = _center_square_crop(raw, frac=0.8)
        if cc is None:
            raise HTTPException(status_code=400, detail=f"no crop ({tag})")
        cropped, tag = cc, "center_crop"
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
