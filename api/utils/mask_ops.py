# api/utils/mask_ops.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional, Union

Array = Union[np.ndarray]

# ---------------------------
# Conversions / binarization
# ---------------------------
def _to_numpy(x: Array) -> np.ndarray:
    """Accepts numpy or torch; returns numpy (cpu)."""
    try:
        import torch  # optional
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def binarize(prob_map: Array, thresh: float = 0.5) -> np.ndarray:
    """
    prob_map: HxW float in [0,1] or any real-valued logits.
    Returns: uint8 mask {0,1}
    """
    p = _to_numpy(prob_map).astype(np.float32)
    # If looks like logits, squash
    if p.min() < 0 and p.max() > 1:
        p = 1.0 / (1.0 + np.exp(-p))
    m = (p >= float(thresh)).astype(np.uint8)
    return m

# ---------------------------
# Cleaning
# ---------------------------
import cv2
import numpy as np

def clean_mask(
    mask: np.ndarray,
    *,
    keep: str = "largest",          # "largest" | "all"
    min_area: int = 800,            # remove tiny blobs before choosing largest
    close_kernel: int = 7,          # odd number; 0 disables closing
    close_iter: int = 1,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Normalize -> denoise -> (optionally) keep only the largest connected component -> smooth -> fill holes.
    Returns a binary mask uint8 {0,1}.
    """
    if mask is None:
        raise ValueError("clean_mask: mask is None")
    m = (mask > 0).astype(np.uint8)

    # --- remove tiny blobs early ---
    if min_area and min_area > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep_ids = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        m = np.isin(lab, keep_ids).astype(np.uint8)

    # --- optionally pick only the largest connected component ---
    if keep == "largest":
        num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num > 1:
            # component 0 is background
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                max_id = 1 + int(np.argmax(areas))
                m = (lab == max_id).astype(np.uint8)

    # --- morphological closing to bridge gaps and smooth edges ---
    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=max(1, close_iter))

    # --- hole filling (keep outer background) ---
    if fill_holes:
        h, w = m.shape[:2]
        flood = m.copy()
        border = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, border, (0, 0), 255)
        inv = cv2.bitwise_not(flood)
        m = cv2.bitwise_or(m * 255, inv) // 255
        m = (m > 0).astype(np.uint8)

    return m



# ---------------------------
# Crop helpers
# ---------------------------
def bbox_from_mask(
    mask_bin: Array,
    pad: float = 0.10,           # relative padding
    img_shape: Optional[Tuple[int,int]] = None,
    square: bool = False
) -> Optional[Tuple[int,int,int,int]]:
    """
    Returns (x0, y0, x1, y1) in pixel coords or None if empty.
    """
    m = _to_numpy(mask_bin).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # padding
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if square:
        side = max(w, h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = int(cx - side/2), int(cx + side/2)
        y0, y1 = int(cy - side/2), int(cy + side/2)
        w = h = side

    pad_x = int(round(w * pad))
    pad_y = int(round(h * pad))
    x0, y0 = x0 - pad_x, y0 - pad_y
    x1, y1 = x1 + pad_x, y1 + pad_y

    # clamp to image size if provided
    if img_shape is not None:
        H, W = img_shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W-1, x1), min(H-1, y1)

    return int(x0), int(y0), int(x1), int(y1)

def crop_from_mask(
    image_bgr: Array,
    mask_bin: Array,
    pad: float = 0.10,
    square: bool = False,
    out_size: Optional[Tuple[int,int]] = None
) -> Tuple[np.ndarray, Optional[Tuple[int,int,int,int]]]:
    """
    Returns cropped image (BGR like input) and bbox.
    If mask is empty, returns original image and None.
    """
    img = _to_numpy(image_bgr)
    bbox = bbox_from_mask(mask_bin, pad=pad, img_shape=img.shape, square=square)
    if bbox is None:
        return img.copy(), None
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1+1, x0:x1+1].copy()
    if out_size:
        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    return crop, bbox

# ---------------------------
# Visualization
# ---------------------------
import cv2
import numpy as np

def overlay_mask_rgb(
    img: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a binary mask onto an RGB image with transparency.
    
    Args:
        img: RGB image (H,W,3) uint8
        mask: Binary mask (H,W), values 0/1
        color: Overlay color (R,G,B)
        alpha: Transparency factor (0=no mask, 1=full mask)
    
    Returns:
        Blended RGB image
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    overlay = img.copy()
    color_arr = np.zeros_like(img, dtype=np.uint8)
    color_arr[:, :] = color

    # Apply only where mask==1
    overlay = np.where(mask[..., None] == 1, color_arr, overlay)

    # Blend original and overlay
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return blended


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Wrapper to overlay a mask on an image."""
    return overlay_mask_rgb(img, mask, color=color, alpha=alpha)

import cv2
import numpy as np

def remove_small_components(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Remove connected components smaller than min_area.
    Args:
        mask: binary uint8 [H,W], values {0,1}
        min_area: minimum pixel area to keep
    Returns:
        cleaned mask with only components >= min_area
    """
    if mask is None or mask.size == 0:
        return mask

    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for lbl in range(1, num_labels):  # skip background
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lbl] = 1

    return cleaned

# --- NEW: component and bbox helpers -----------------------------------------
import numpy as np
import cv2

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    mask: np.uint8 {0,1} or {0,255}
    returns binary mask {0,1}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # normalize to {0,1}
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:  # background + at most one component
        return m

    # stats[:, cv2.CC_STAT_AREA] -> pick largest non-background (label 0 is bg)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = (labels == largest_idx).astype(np.uint8)
    return out


def _pad_bbox(x1, y1, x2, y2, H, W, pad_frac: float = 0.08):
    """Pad a bbox by pad_frac of its size, then clip to image bounds."""
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    pad_w = int(round(w * pad_frac))
    pad_h = int(round(h * pad_frac))
    x1p = max(0, x1 - pad_w); y1p = max(0, y1 - pad_h)
    x2p = min(W - 1, x2 + pad_w); y2p = min(H - 1, y2 + pad_h)
    return x1p, y1p, x2p, y2p


def crop_from_mask_refined(
    img_rgb: np.ndarray,
    mask_bin: np.ndarray,
    *,
    padding: float = 0.08,
    min_area: int = 400,
    max_aspect_ratio: float = 5.0,
    fallback_size: int | None = 224,
):
    """
    Robust crop extraction:
      1) keep largest component
      2) discard tiny masks
      3) bbox with padding + clipping
      4) reject extreme aspect ratios and optionally center-crop as fallback

    Returns (crop_rgb, bbox) where bbox=(x1,y1,x2,y2). If no crop is possible,
    returns (img_rgb, None) or a center-crop if fallback_size is set.
    """
    H, W = img_rgb.shape[:2]

    m = (mask_bin > 0).astype(np.uint8)
    if m.sum() == 0:
        # fallback
        if fallback_size is None:
            return img_rgb, None
        # center crop square
        s = min(fallback_size, H, W)
        cx, cy = W // 2, H // 2
        x1 = max(0, cx - s // 2); y1 = max(0, cy - s // 2)
        x2 = min(W, x1 + s);      y2 = min(H, y1 + s)
        return img_rgb[y1:y2, x1:x2], (x1, y1, x2 - 1, y2 - 1)

    m = keep_largest_component(m)

    # Remove very small largest-components
    if int(m.sum()) < min_area:
        if fallback_size is None:
            return img_rgb, None
        s = min(fallback_size, H, W)
        cx, cy = W // 2, H // 2
        x1 = max(0, cx - s // 2); y1 = max(0, cy - s // 2)
        x2 = min(W, x1 + s);      y2 = min(H, y1 + s)
        return img_rgb[y1:y2, x1:x2], (x1, y1, x2 - 1, y2 - 1)

    ys, xs = np.where(m > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # pad + clip
    x1, y1, x2, y2 = _pad_bbox(x1, y1, x2, y2, H, W, pad_frac=padding)

    # sanity: aspect ratio
    bw = max(1, x2 - x1 + 1); bh = max(1, y2 - y1 + 1)
    ar = max(bw / bh, bh / bw)
    if ar > max_aspect_ratio:
        if fallback_size is None:
            return img_rgb, (x1, y1, x2, y2)
        s = min(fallback_size, H, W)
        cx, cy = W // 2, H // 2
        fx1 = max(0, cx - s // 2); fy1 = max(0, cy - s // 2)
        fx2 = min(W, fx1 + s);     fy2 = min(H, fy1 + s)
        return img_rgb[fy1:fy2, fx1:fx2], (fx1, fy1, fx2 - 1, fy2 - 1)

    crop = img_rgb[y1:y2+1, x1:x2+1]
    return crop, (x1, y1, x2, y2)

# --- Add to api/utils/mask_ops.py -------------------------------------------
import numpy as _np
import cv2 as _cv2

def keep_largest_component(mask: _np.ndarray) -> _np.ndarray:
    """
    Keep only the largest connected component of a binary mask.
    mask: uint8 {0,1} or {0,255}
    """
    if mask.dtype != _np.uint8:
        mask = mask.astype(_np.uint8)
    if mask.max() == 255:
        work = (mask > 0).astype(_np.uint8)
    else:
        work = mask.copy()

    num, lbl = _cv2.connectedComponents(work, connectivity=4)
    if num <= 2:  # background + at most one blob
        return work

    best_label, best_area = 0, 0
    for l in range(1, num):
        a = int((lbl == l).sum())
        if a > best_area:
            best_area, best_label = a, l
    return (lbl == best_label).astype(_np.uint8)


def refine_mask_from_prob(
    prob: _np.ndarray,
    hard_thresh: float = 0.50,
    tiny_ratio: float = 0.005,     # if <0.5% pixels on, try softer threshold
    fallback_pct: float = 94.0,    # percentile used for fallback threshold
    min_area_ratio: float = 0.002, # drop blobs <0.2% of image
    close_kernel: int = 5,         # morphological close to bridge gaps
) -> _np.ndarray:
    """
    Convert a probability map [0..1] to a stable binary mask:
    - hard threshold (0.5)
    - if area is tiny, use a percentile fallback
    - clean with morphology + fill from clean_mask()
    - remove small components, keep largest
    Returns uint8 mask {0,1}
    """
    H, W = prob.shape[:2]
    raw = (prob > hard_thresh).astype(_np.uint8)

    # fallback if too small
    if raw.sum() < tiny_ratio * (H * W):
        t = _np.percentile(prob, fallback_pct)
        raw = (prob > t).astype(_np.uint8)

    # morphology close to smooth/bridge
    if close_kernel and close_kernel > 1:
        k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        raw = _cv2.morphologyEx(raw, _cv2.MORPH_CLOSE, k)

    # your existing cleaner (handles fill/holes etc.)
    cleaned = clean_mask(raw)

    # drop very small blobs
    min_area = int(min_area_ratio * (H * W))
    if min_area > 0:
        cleaned = remove_small_components(cleaned, min_area=min_area)

    # keep the biggest region to stabilize crops
    cleaned = keep_largest_component(cleaned)
    return cleaned.astype(_np.uint8)


def crop_from_mask_with_pad(
    img_rgb: _np.ndarray,
    mask: _np.ndarray,
    pad: float = 0.12,
    min_side: int = 80,
) -> _np.ndarray:
    """
    Crop img around the mask bbox with proportional padding.
    Returns a cropped RGB array; if mask empty, returns zero-sized array.
    """
    if mask.dtype != _np.uint8:
        mask = mask.astype(_np.uint8)
    m = (mask > 0).astype(_np.uint8)
    ys, xs = _np.where(m > 0)
    if xs.size == 0:
        return _np.zeros((0, 0, 3), dtype=_np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = img_rgb.shape[:2]

    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)
    # ensure minimum box size before padding
    if bw < min_side:
        delta = (min_side - bw) // 2
        x1 -= delta; x2 += delta
    if bh < min_side:
        delta = (min_side - bh) // 2
        y1 -= delta; y2 += delta

    # add proportional padding
    pw = int(pad * (x2 - x1 + 1))
    ph = int(pad * (y2 - y1 + 1))
    x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
    x2 = min(w - 1, x2 + pw); y2 = min(h - 1, y2 + ph)

    return img_rgb[y1:y2+1, x1:x2+1].copy()
# ---------------------------------------------------------------------------

