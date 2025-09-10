# api/routes/anemia.py
import os, io, time, hashlib, traceback
from pathlib import Path
from typing import Optional, Tuple

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps

# Optional cropper deps
import cv2
import numpy as np

from api.utils.config import get_anemia_threshold
from training.anemia.train import build_model  # used when loading resnet18 ckpts

# Try to import pseudomask helper (optional)
try:
    from ..psuedomask import run_pseudomask_on_bytes  # your existing util
except Exception:
    run_pseudomask_on_bytes = None

router = APIRouter()

# -------- Device & determinism --------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------- Config --------
CKPT_PATH = Path(os.getenv("ANEMIA_CKPT", ""))      # REQUIRED at runtime
IMG_SIZE = int(os.getenv("ANEMIA_IMG_SIZE", "224"))
MAX_IMAGE_BYTES = int(os.getenv("ANEMIA_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))  # 5MB default
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
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -------- Checkpoint helpers --------
def _extract_state_dict(obj):
    """Accept {"model": sd}, {"state_dict": sd}, or raw state_dict."""
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
    return "resnet18"

def _remap_simplecnn_linear(sd: dict) -> dict:
    """
    If the checkpoint stored the final Linear layer under 'cnn.<i>.(weight|bias)'
    (2-D weight), remap it to 'fc.(weight|bias)' so our module can load it.
    """
    sd2 = dict(sd)
    # find any 2-D cnn.* weight (Linear)
    linear_keys = [k for k in sd.keys() if k.startswith("cnn.") and k.endswith(".weight") and sd[k].ndim == 2]
    if not linear_keys:
        return sd2
    # pick the highest index (likely the last layer)
    lk = max(linear_keys, key=lambda k: int(k.split(".")[1]))
    li = int(lk.split(".")[1])
    w = sd[lk]
    b = sd.get(f"cnn.{li}.bias", None)
    # place into fc.*
    sd2["fc.weight"] = w
    if b is not None:
        sd2["fc.bias"] = b
    # remove old entries to avoid "unexpected keys"
    del sd2[lk]
    if b is not None and f"cnn.{li}.bias" in sd2:
        del sd2[f"cnn.{li}.bias"]
    return sd2

# -------- Dynamic SimpleCNN (matches cnn.* convs; fc is separate) --------
class SimpleCNNDynamic(nn.Module):
    """
    Builds a 'cnn' stack from checkpoint keys:
      - 4D weights -> Conv2d at that index in a Sequential (gaps = Identity)
      - 2D weights (Linear) are NOT placed in cnn; they become fc.* via remap
    If the last conv channels != fc.in_features, a 1x1 'align' conv is added.
    """
    def __init__(self, sd, num_classes_default=2):
        super().__init__()
        # Collect 4D conv weights
        conv_w = {
            int(k.split(".")[1]): sd[k]
            for k in sd.keys()
            if k.startswith("cnn.") and k.endswith(".weight") and sd[k].ndim == 4
        }
        if not conv_w:
            raise RuntimeError("simple_cnn: no 4D conv weights found in checkpoint")

        max_idx = max(conv_w.keys())
        seq = []
        last_out = None
        for i in range(max_idx + 1):
            if i in conv_w:
                w = conv_w[i]
                out_ch, in_ch, kh, kw = w.shape
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw),
                                 padding=(kh // 2, kw // 2), bias=True)
                seq.append(conv)
                last_out = int(out_ch)
            else:
                seq.append(nn.Identity())
        self.cnn = nn.Sequential(*seq)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.align = None
        # fc will be created in _finalize_fc(...)
        self.fc = None
        self._last_out = last_out
        self._num_classes_default = int(num_classes_default)

    def _finalize_fc(self, fc_in: Optional[int], fc_out: Optional[int]):
        if fc_in is not None and fc_out is not None:
            # add align if channels don't match
            if self._last_out is None:
                raise RuntimeError("simple_cnn: cannot infer channels before fc")
            if self._last_out != fc_in:
                self.align = nn.Conv2d(self._last_out, fc_in, kernel_size=1, bias=True)
            self.fc = nn.Linear(fc_in, fc_out)
        else:
            # default small head if fc not in checkpoint
            if self._last_out is None:
                raise RuntimeError("simple_cnn: cannot infer channels for classifier")
            self.fc = nn.Linear(self._last_out, self._num_classes_default)

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        if self.align is not None:
            x = self.align(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------- Loader --------
def _load_classifier_strict():
    """
    Loads either ResNet-18 or SimpleCNN (auto-detected by checkpoint keys,
    unless ANEMIA_ARCH overrides). Populates globals; raises on fatal issues.
    """
    global _classifier, _xform, _classes, _backbone, _model_tag

    ckpt_path = str(CKPT_PATH)
    if not ckpt_path:
        raise RuntimeError("ANEMIA_CKPT not set.")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Classifier checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=_device)
    sd = _extract_state_dict(obj)

    # Optional metadata
    if isinstance(obj, dict):
        meta_classes = obj.get("classes")
        if meta_classes and isinstance(meta_classes, (list, tuple)) and "anemic" in meta_classes:
            _classes = list(meta_classes)

    arch = ANEMIA_ARCH if ANEMIA_ARCH != "auto" else _infer_arch_from_keys(sd.keys())

    if arch == "resnet18":
        model = build_model(num_classes=2, backbone="resnet18", pretrained=False).to(_device)
        # strict load; will raise if mismatch
        model.load_state_dict(sd, strict=True)
        _backbone = "resnet18"
    elif arch == "simple_cnn":
        # Inspect for a 2-D linear under cnn.* to size fc
        linear_keys = [k for k in sd.keys() if k.startswith("cnn.") and k.endswith(".weight") and sd[k].ndim == 2]
        fc_in = fc_out = None
        if linear_keys:
            lk = max(linear_keys, key=lambda k: int(k.split(".")[1]))
            fc_out, fc_in = sd[lk].shape  # (out_features, in_features)

        model = SimpleCNNDynamic(sd).to(_device)
        model._finalize_fc(fc_in, fc_out)

        # Remap cnn.<i> (2D) -> fc.* so we actually load the head weights
        sd2 = _remap_simplecnn_linear(sd)

        missing, unexpected = model.load_state_dict(sd2, strict=False)
        if missing:
            print(f"[ISHI] simple_cnn missing keys: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[ISHI] simple_cnn unexpected keys: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")
        _backbone = "simple_cnn"
    else:
        raise RuntimeError(f"Unknown ANEMIA_ARCH: {arch}")

    model.eval()
    _classifier = model
    _xform = _make_transform()
    _model_tag = os.path.basename(ckpt_path) or arch
    print(f"[ISHI] Classifier ready: tag={_model_tag} arch={_backbone} classes={_classes}")
    return _classifier, _xform

def preload_model():
    """Called from app startup to fail fast if model can't load."""
    return _load_classifier_strict()

# -------- Inference --------
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

# -------- Route --------
@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Prediction endpoint (auto-detects arch from ckpt if ANEMIA_ARCH=auto):
    - field: 'image' (multipart/form-data)
    """
    if _classifier is None or _xform is None:
        raise HTTPException(status_code=503, detail="model not ready")

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
    
# in api/routes/anemia.py
from fastapi.responses import Response

@router.post("/debug/crop")
async def debug_crop(image: UploadFile = File(...)):
    raw = await image.read()
    cropped, tag = _try_mask_crop(raw)
    if cropped is None:
        cc = _center_square_crop(raw, frac=0.8)
        if cc is None:
            raise HTTPException(status_code=400, detail=f"no crop ({tag})")
        cropped, tag = cc, "center_crop"
    import io
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

