# /workspaces/ishi/api/anemia_classifier.py
"""
Binary mask -> crop -> classify with your model:
  /workspaces/ishi/models/anemia/anemia_cnn.pth

Load order:
1) torch.jit.load (TorchScript)
2) torch.load (full model)
3) state_dict -> torchvision resnet18 head (1 logit).
   Replace build_fallback_model() with your net if needed.

Returns: {"label": "Anemic"|"Not Anemic", "probability": float, "input_size": int, [b64 images]}
"""

import os
import base64
from typing import Dict, Any, Optional, TYPE_CHECKING

import cv2
import numpy as np

# --- Optional runtime torch imports (may be absent in some envs) ---
try:
    import torch  # runtime handle (may be None if import fails)
    from torchvision import models, transforms
except Exception:
    torch = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore

# --- Static typing (for Pylance/mypy) without importing torch at runtime ---
if TYPE_CHECKING:
    import torch as Torch
    from torch import Tensor as TorchTensor
else:
    Torch = Any  # type: ignore
    TorchTensor = Any  # type: ignore

# ----- CONFIG -----
DEFAULT_MODEL_PATH = os.environ.get(
    "ANEMIA_MODEL_PATH",
    "/workspaces/ishi/models/anemia/anemia_cnn.pth"  # <<-- your file
)
INPUT_SIZE = int(os.environ.get("ANEMIA_MODEL_INPUT", "224"))
MODEL_DEVICE = os.environ.get("ANEMIA_MODEL_DEVICE", "").strip().lower()  # "", "cpu", "cuda"
USE_SOFTMAX = False  # set True if your model outputs 2 logits [not_anemic, anemic]

# ----- UTILS -----
def _enc_png(img_u8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def _b64_png(img_u8: np.ndarray) -> str:
    return base64.b64encode(_enc_png(img_u8)).decode("ascii")

def _apply_mask_and_crop(bgr: np.ndarray, mask_u8: np.ndarray, pad: int = 6) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return bgr
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr
    largest = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = m.shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return bgr[y0:y1, x0:x1].copy()

def _to_rgb_resized(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

# ----- MODEL LOAD -----
_model_cache: Optional["Torch.nn.Module"] = None  # <- safe for Pylance due to TYPE_CHECKING above

def _pick_device() -> str:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    if MODEL_DEVICE in ("cpu", "cuda"):
        return MODEL_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_fallback_model(num_outputs: int = 1) -> "Torch.nn.Module":
    """
    Fallback net if checkpoint is a state_dict. Replace with your architecture if needed.
    """
    if models is None:
        raise RuntimeError("torchvision not installed.")
    m = models.resnet18(weights=None)
    m.fc = __import__("torch").nn.Linear(m.fc.in_features, num_outputs)  # late-binding to avoid type warnings
    return m

def _try_torchscript(path: str, device: str):
    try:
        m = __import__("torch").jit.load(path, map_location=device)
        m.eval().to(device)
        return m
    except Exception:
        return None

def _try_full(path: str, device: str):
    try:
        m = __import__("torch").load(path, map_location=device)
        if not hasattr(m, "forward"):
            return None
        m.eval().to(device)
        return m
    except Exception:
        return None

def _try_state_dict(path: str, device: str):
    try:
        sd = __import__("torch").load(path, map_location="cpu")
        if not isinstance(sd, dict):
            return None
        m = build_fallback_model(num_outputs=1)  # change to 2 + USE_SOFTMAX=True if needed
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        return m
    except Exception:
        return None

def load_model(path: str = DEFAULT_MODEL_PATH) -> "Torch.nn.Module":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    dev = _pick_device()
    for loader in (_try_torchscript, _try_full, _try_state_dict):
        m = loader(path, dev)
        if m is not None:
            _model_cache = m
            return m
    raise RuntimeError(
        f"Failed to load {path}. If it's a state_dict for a custom net, "
        f"replace build_fallback_model() with your architecture."
    )

# ----- INFERENCE -----
def _preprocess(rgb_uint8: np.ndarray) -> "TorchTensor":
    if transforms is None:
        raise RuntimeError("torchvision not installed.")
    t = transforms.ToTensor()(rgb_uint8)
    t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])(t)
    return t.unsqueeze(0)

def classify_crop(bgr: np.ndarray, mask_u8: np.ndarray) -> Dict[str, Any]:
    model = load_model(DEFAULT_MODEL_PATH)
    crop_bgr = _apply_mask_and_crop(bgr, mask_u8, pad=6)
    rgb = _to_rgb_resized(crop_bgr, INPUT_SIZE)
    x = _preprocess(rgb).to(next(model.parameters()).device)

    with __import__("torch").inference_mode():
        out = model(x)

    if USE_SOFTMAX:
        # Expect [1,2] logits => softmax -> index 1 is Anemic
        if out.ndim == 2 and out.shape[1] == 2:
            prob = __import__("torch").softmax(out, dim=1)[0, 1].item()
        else:
            raise RuntimeError("Model output not [1,2] while USE_SOFTMAX=True.")
    else:
        logit = out.view(-1)[0].item()
        prob = 1.0 / (1.0 + np.exp(-logit))

    label = "Anemic" if prob >= 0.5 else "Not Anemic"
    return {"label": label, "probability": float(prob), "input_size": int(INPUT_SIZE)}

def analyze_with_mask(bgr: np.ndarray, mask_u8: np.ndarray, include_images_b64: bool = True) -> Dict[str, Any]:
    # overlay
    color = (0, 165, 255)
    color_img = np.zeros_like(bgr); color_img[:] = np.array(color, np.uint8)
    m3 = cv2.merge([mask_u8, mask_u8, mask_u8]) // 255
    blend = (0.45 * color_img + 0.55 * bgr).astype(np.uint8)
    overlay = (m3 * blend) + ((1 - m3) * bgr)

    # crop + classify
    crop_bgr = _apply_mask_and_crop(bgr, mask_u8, pad=6)
    clf = classify_crop(bgr, mask_u8)

    out: Dict[str, Any] = {**clf}
    if include_images_b64:
        out["mask_png_b64"] = _b64_png(mask_u8)
        out["overlay_png_b64"] = _b64_png(overlay)
        out["crop_png_b64"] = _b64_png(crop_bgr)
    return out
