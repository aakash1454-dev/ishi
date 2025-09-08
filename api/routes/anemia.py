# /workspaces/ishi/api/routes/anemia.py
import io, os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

router = APIRouter(tags=["anemia"])  # no prefix -> route will be /predict

ANEMIA_CKPT = os.getenv("ANEMIA_CKPT", "runs/anemia_resnet18_v1/best_acc.pth")
CROP_CKPT   = os.getenv("CROP_CKPT",   "runs/unet_crop/unet_crop_best.pth")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Optional cropper ----------
class _IdentityCrop:
    def __call__(self, img_pil: Image.Image) -> Image.Image:
        return img_pil

def _try_load_cropper():
    # If ckpt missing, skip
    if not os.path.isfile(CROP_CKPT):
        return _IdentityCrop(), "cropper_skipped_no_ckpt"

    # Try to import UNet from a refactored module first, then fallback to train file.
    net = None
    reason = ""
    try:
        try:
            from models.anemia.cropping.unet import UNet  # preferred
        except Exception:
            # Fallback: train script defines UNet (your repo note says it’s embedded here)
            from models.anemia.cropping.train_unet_crop import UNet  # may fail if not exported
        net = UNet()
        state = torch.load(CROP_CKPT, map_location=_device)
        # Support both raw state_dict or checkpoints with 'model' key
        sd = state.get("model", state)
        net.load_state_dict(sd, strict=False)
        net.to(_device).eval()
    except Exception as e:
        return _IdentityCrop(), f"cropper_failed:{type(e).__name__}"

    # basic center-mask crop using UNet output
    import numpy as np
    import torchvision.transforms.functional as TF

    class _UNetCrop:
        def __call__(self, img_pil: Image.Image) -> Image.Image:
            img = img_pil.convert("RGB")
            t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
            x = t(img).unsqueeze(0).to(_device)
            with torch.no_grad():
                m = net(x)  # assume shape [1,1,H,W] or [1,2,H,W]
                if m.shape[1] > 1:
                    m = torch.softmax(m, dim=1)[:, 1:2]  # foreground
                m = (m > 0.5).float()[0, 0].cpu().numpy()  # HxW mask
            # Find bbox of mask; fallback to full image if empty
            ys, xs = np.where(m > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                return img  # no crop if mask empty
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            # Map bbox from 256-resized back to original
            H, W = img.size[1], img.size[0]  # PIL: (W,H)
            scale_x = W / 256.0
            scale_y = H / 256.0
            box = (int(x0 * scale_x), int(y0 * scale_y), int((x1 + 1) * scale_x), int((y1 + 1) * scale_y))
            return img.crop(box)

    return _UNetCrop(), "cropper_ok"

# ---------- Classifier ----------
def _load_classifier():
    import torchvision.models as models
    import torch
    import torch.nn as nn
    import os

    ckpt_path = os.getenv("ANEMIA_CKPT", "runs/anemia_resnet18_v1/best_acc.pth")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Classifier checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=_device)
    sd = state.get("model", state)  # support {'model': state_dict} or raw state_dict

    # --- candidate model builders (2-class heads) ---
    def resnet18():
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m

    def mobilenet_v3_small():
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 2)
        return m

    def mobilenet_v3_large():
        m = models.mobilenet_v3_large(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 2)
        return m

    def efficientnet_b0():
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 2)
        return m

    # Tiny fallback CNN if you trained a simple custom model
    class TinyCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.avg = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.features(x)
            x = self.avg(x).flatten(1)
            return self.classifier(x)

    candidates = [
        ("resnet18", resnet18),
        ("mobilenet_v3_small", mobilenet_v3_small),
        ("mobilenet_v3_large", mobilenet_v3_large),
        ("efficientnet_b0", efficientnet_b0),
        ("tinycnn", TinyCNN),
    ]

    # scoring function: more matching keys with same shapes = better
    def score(model, sd):
        m_sd = model.state_dict()
        match = 0
        shape_match = 0
        for k, v in sd.items():
            if k in m_sd:
                match += 1
                if tuple(m_sd[k].shape) == tuple(v.shape):
                    shape_match += 1
        # prioritize exact shape matches, then total matches
        return (shape_match, match)

    best = None
    best_name = None
    best_score = (-1, -1)

    for name, ctor in candidates:
        try:
            m = ctor()
            s = score(m, sd)
            if s > best_score:
                best, best_name, best_score = m, name, s
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Could not match checkpoint to any known model.")

    # load with strict=False (head or minor keys may differ)
    missing, unexpected = best.load_state_dict(sd, strict=False)
    # You can print/log these if ISHI_DEBUG is on:
    # print(f"Picked {best_name}; missing={len(missing)} unexpected={len(unexpected)} score={best_score}")

    best.to(_device).eval()

    import torchvision.transforms as T
    xform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return best, xform


_classifier, _xform = None, None
_cropper = None
_cropper_status = "unset"

@router.post("/predict")  # final path: /predict
async def predict(image: UploadFile = File(...)):
    global _classifier, _xform, _cropper, _cropper_status
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # lazy-load
    if _cropper is None:
        _cropper, _cropper_status = _try_load_cropper()

    if _classifier is None:
        _classifier, _xform = _load_classifier()

    # crop (or identity)
    img_c = _cropper(img)

    # classify
    x = _xform(img_c).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _classifier(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    score_anemic = float(probs[1])  # assuming class 1 = anemic
    pred = bool(score_anemic >= 0.5)

    return JSONResponse({
        "anemic": pred,
        "score": score_anemic,
        "cropper": _cropper_status,
        "model": "anemia_resnet18_v1",
    })
