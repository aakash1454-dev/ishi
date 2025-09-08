# scripts/spotcheck_masks.py
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.crop.train_unet import UNetSmall, EyelidSegDataset, DEVICE
from api.utils.mask_ops import (
    clean_mask,
    remove_small_components,
    overlay_mask,
    keep_largest_component,
    crop_from_mask_refined,
)

# --- paths ---
ROOTS = [
    Path("/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia/India"),
    Path("/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia/Italy"),
]
CKPT_PATH = Path("models/crop/unet_crop_best.pth")
OUT_DIR = Path("models/crop/spotcheck"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_uint8_rgb(t3):
    # t3: torch tensor [3,H,W] in 0..1
    arr = (t3.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr

def draw_bbox(img_rgb, bbox, color=(255, 0, 0), thickness=2):
    if bbox is None:
        return img_rgb
    x1, y1, x2, y2 = map(int, bbox)
    img = img_rgb.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# --- load model ---
model = UNetSmall().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

# --- dataset ---
ds = EyelidSegDataset(ROOTS)
print(f"Found {len(ds)} pairs.")

# --- run a few samples ---
for i in [0, 5, 42, 100, 200]:
    img_t, _ = ds[i]  # [3,H,W] tensor
    img_np = to_uint8_rgb(img_t)

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(DEVICE))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [H,W] float

    # 1) initial hard threshold
    raw = (prob > 0.5).astype(np.uint8)

    # 2) fallback to softer percentile if too tiny (<0.5% pixels)
    if raw.sum() < 0.005 * prob.size:
        t = np.percentile(prob, 94)
        raw = (prob > t).astype(np.uint8)

    # 3) clean + remove blobs + keep largest
    mask_clean = clean_mask(raw)
    mask_clean = remove_small_components(mask_clean, min_area=200)
    mask_clean = keep_largest_component(mask_clean)

    # 4) overlay
    overlay = overlay_mask(img_np, mask_clean, alpha=0.4)

    # 5) refined crop with padding + fallbacks
    crop_img, bbox = crop_from_mask_refined(
        img_np, mask_clean,
        padding=0.08,
        min_area=400,
        max_aspect_ratio=5.0,
        fallback_size=224,
    )

    # 6) draw bbox on original
    bbox_vis = draw_bbox(img_np, bbox, color=(255, 0, 0), thickness=2)

    # 7) save
    cv2.imwrite(str(OUT_DIR / f"check_{i:04d}.png"),         cv2.cvtColor(img_np,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / f"check_{i:04d}_overlay.png"), cv2.cvtColor(overlay,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / f"check_{i:04d}_bbox.png"),    cv2.cvtColor(bbox_vis,  cv2.COLOR_RGB2BGR))
    if isinstance(crop_img, np.ndarray) and crop_img.size > 0:
        cv2.imwrite(str(OUT_DIR / f"check_{i:04d}_crop.png"), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

print(f"✅ Spot check results saved to {OUT_DIR}")
