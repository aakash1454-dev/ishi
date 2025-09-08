# api/routes/anemia.py
import os, io, time, hashlib
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from api.utils.config import get_anemia_threshold

# Your training build helper (adjust if your project path differs)
from training.anemia.train import build_model

router = APIRouter()

# -------------------------
# Device & deterministic flags
# -------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# Config (env or sane defaults)
# -------------------------
CKPT_PATH = Path(os.getenv("ANEMIA_CKPT", "runs/anemia_resnet18_v1/best_acc.pth"))
BACKBONE_ENV = os.getenv("ANEMIA_BACKBONE", "")  # if empty, infer from ckpt or default
IMG_SIZE = int(os.getenv("ANEMIA_IMG_SIZE", "224"))

# -------------------------
# Globals (lazy-init)
# -------------------------
_classifier = None
_xform = None
_classes = ["anemic", "nonanemic"]  # canonical default order
_backbone = BACKBONE_ENV or "resnet18"
_model_tag = "anemia_resnet18_v1"

# -------------------------
# Helpers
# -------------------------
def _make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def _extract_state_dict(obj):
    """
    Accept:
      - {"model": state_dict}
      - {"state_dict": state_dict}
      - raw state_dict (param_name -> tensor)
    """
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # Heuristic: looks like a raw state_dict
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError(
        f"Unrecognized checkpoint format. Top-level type/keys: "
        f"{type(obj)} / {list(obj.keys()) if isinstance(obj, dict) else 'n/a'}"
    )

def _make_stub_model():
    """Fixed-probability stub so /predict never 500s during UI dev."""
    import math
    import torch.nn as nn
    p = float(os.getenv("ISHI_STUB_SCORE", "0.154"))  # P(anemic)
    eps = 1e-6
    logit = math.log(max(min(p, 1 - eps), eps) / max(1 - p, eps))

    class Stub(nn.Module):
        def forward(self, x):
            b = x.size(0)
            return torch.stack([
                torch.zeros(b, device=x.device),          # logit for "nonanemic"
                torch.full((b,), logit, device=x.device)  # logit for "anemic"
            ], dim=1)

    return Stub().to(_device).eval()

def _load_classifier():
    """
    Returns (model.eval(), transform). Uses:
      - ANEMIA_CKPT env var (preferred)
      - else default "runs/anemia_resnet18_v1/best_acc.pth"
    If missing and ISHI_ALLOW_STUB=1 -> returns stub model.
    Also sets global _classes, _backbone, _model_tag if present in ckpt.
    """
    global _classifier, _xform, _classes, _backbone, _model_tag

    if _classifier is not None and _xform is not None:
        return _classifier, _xform

    ckpt_path = str(CKPT_PATH)
    allow_stub = os.getenv("ISHI_ALLOW_STUB", "0") == "1"

    if os.path.isfile(ckpt_path):
        print(f"[ISHI] Loading classifier ckpt: {ckpt_path}")
        obj = torch.load(ckpt_path, map_location=_device)

        # metadata (optional)
        if isinstance(obj, dict):
            meta_backbone = obj.get("backbone")
            meta_classes = obj.get("classes")
            if meta_classes and isinstance(meta_classes, (list, tuple)) and "anemic" in meta_classes:
                _classes = list(meta_classes)
            if BACKBONE_ENV:
                _backbone = BACKBONE_ENV
            elif meta_backbone:
                _backbone = meta_backbone
            else:
                _backbone = _backbone or "resnet18"

        # Build model & load weights
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
        return _classifier, _xform

    # No checkpoint on disk
    if not allow_stub:
        raise RuntimeError(f"Classifier checkpoint not found: {ckpt_path}")

    print(f"[ISHI] No ckpt at {ckpt_path}; using STUB (ISHI_ALLOW_STUB=1).")
    _classifier = _make_stub_model()
    _xform = _make_transform()
    _model_tag = "stub"
    return _classifier, _xform

def _predict_p_anemic(pil_img: Image.Image) -> float:
    model, xform = _load_classifier()
    x = xform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # pick index for "anemic" from known classes
    try:
        idx_anemic = _classes.index("anemic")
    except ValueError:
        idx_anemic = 1  # sensible default if classes missing
    return float(probs[idx_anemic])

# -------------------------
# Routes
# -------------------------
@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    CNN-based anemia prediction on uploaded image.
    - field name: 'image' (multipart/form-data)
    - returns: { anemic, score, cropper, model, threshold, ... }
    """
    t0 = time.time()
    try:
        raw = await image.read()

        # EXIF-safe open and convert to RGB
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")

        # inference
        p_anemic = _predict_p_anemic(img)

        # threshold (config), default 0.50
        threshold = float(get_anemia_threshold() or 0.50)
        is_anemic = p_anemic >= threshold

        return JSONResponse({
            "anemic": bool(is_anemic),
            "score": float(p_anemic),
            "cropper": "cropper_skipped_no_ckpt",   # update when cropper is wired
            "model": _model_tag,
            "backbone": _backbone,
            "classes": _classes,
            "threshold": threshold,
            "latency_ms": int((time.time() - t0) * 1000),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    except Exception as e:
        # Surface as 500 so the client knows this is server-side
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")
