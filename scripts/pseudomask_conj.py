#!/usr/bin/env python3
import os, sys, cv2, numpy as np

ALPHA = 0.35
COLOR_PAL = (0, 85, 255)   # BGR coral/red-orange
COLOR_FRX = (255, 255, 0)  # BGR orange
PAD = 8
PADDING = 24  # pixels of reflective padding around the image
def preprocess_img(img_bgr):
    """CLAHE on L channel + simple gray-world white-balance to reduce domain gap."""
    # LAB CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    lab_c = cv2.merge([Lc, a, b])
    img = cv2.cvtColor(lab_c, cv2.COLOR_LAB2BGR)

    # gray-world WB
    means = img.reshape(-1,3).mean(axis=0) + 1e-6
    g = means.mean()
    gain = g / means
    img = np.clip(img * gain, 0, 255).astype(np.uint8)
    return img

def pad_img(img):
    return cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_REFLECT_101)

def unpad_mask(mask01):
    return mask01[PADDING:-PADDING, PADDING:-PADDING]


def keep_largest(mask01):
    n, labels = cv2.connectedComponents(mask01.astype(np.uint8))
    if n <= 1: return mask01
    best = 0; best_sz = 0
    for i in range(1, n):
        sz = (labels == i).sum()
        if sz > best_sz: best, best_sz = i, sz
    return (labels == best).astype(np.uint8)

def clean(mask01, k=5):
    k = np.ones((k,k), np.uint8)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, k)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, k)
    mask01 = keep_largest(mask01)
    return mask01

def redness_mask(img_bgr):
    # Convert to CIE Lab and threshold "a*" (red‑green axis)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    # Heuristic thresholds; tune if needed
    a_th = int(np.clip(np.median(a) + 6, 125, 200))
    L_low, L_high = 30, 240
    red = ((a >= a_th) & (L >= L_low) & (L <= L_high)).astype(np.uint8)
    return red
   
def sclera_mask(img_bgr):
    # bright + low-red area → sclera (robust to lighting)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    Lm, am = np.median(L), np.median(a)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    glare = (V > 240).astype(np.uint8)     # remove pure specular

    # bright and not too red
    scl = ((L >= Lm) & (a <= am - 3)).astype(np.uint8)
    scl[glare == 1] = 0
    scl = clean(scl, 3)
    return scl



def palpebral_from_redness(img_bgr, a_delta=6, lower_bias_frac=1/3):
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a_th = int(np.clip(np.median(a) + a_delta, 120, 200))
    red = ((a >= a_th) & (L >= 30) & (L <= 240)).astype(np.uint8)
    lower = np.zeros_like(red); lower[int(h*lower_bias_frac):, :] = 1
    cand = (red & lower).astype(np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < np.percentile(gray, 25)).astype(np.uint8)
    cand[dark == 1] = 0
    pal = clean(cand, 7)
    scl = sclera_mask(img_bgr)
    near_scl = cv2.dilate(scl, np.ones((7,7), np.uint8), 1)
    pal[(scl == 1) & (near_scl == 0)] = 0
    pal = cv2.morphologyEx(pal, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    pal = clean(pal, 5)
    return pal

def fornix_from_palpebral(img_bgr, pal01, width_px=16, min_pix=1000, prefer_bright_lowred=True):
    pal01 = (pal01 > 0).astype(np.uint8)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_px, width_px))
    dilated = cv2.dilate(pal01, k_out, iterations=1)
    ring_outside = ((dilated > 0) & (pal01 == 0)).astype(np.uint8)

    edge = cv2.morphologyEx(pal01, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge_dil = cv2.dilate(edge, np.ones((width_px, width_px), np.uint8), iterations=1)
    ring_outside &= edge_dil

    if prefer_bright_lowred:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, a, _ = cv2.split(lab)
        mask_pref = ((L > np.median(L)) & (a < np.median(a))).astype(np.uint8)
        frx = (ring_outside & mask_pref).astype(np.uint8)
    else:
        frx = ring_outside

    if frx.sum() < min_pix:
        frx = ring_outside

    frx = cv2.morphologyEx(frx, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    frx = cv2.morphologyEx(frx, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return frx


def fornix_edge_band(img_bgr, pal01, width_px=10):
    """
    Fallback: derive fornix as a thin band OUTSIDE the palpebral boundary.
    Bias toward bright (sclera-like) pixels, but never returns empty.
    """
    h, w = img_bgr.shape[:2]
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_px, width_px))
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, width_px//2), max(3, width_px//2)))

    # Outside ring around palpebral edge
    dil = cv2.dilate(pal01, k_out, iterations=1)
    ring_outside = ((dil > 0) & (pal01 == 0)).astype(np.uint8)

    # Prefer bright / low-red pixels (sclera-ish), but don't require it
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, _ = cv2.split(lab)
    Lm, am = np.median(L), np.median(a)
    bright_lowred = ((L >= Lm) & (a <= am)).astype(np.uint8)

    frx = (ring_outside & bright_lowred).astype(np.uint8)

    # If still empty, just use the ring_outside as a fallback
    if frx.max() == 0:
        frx = ring_outside

    # Keep near the edge (thin band), and clean
    edge = cv2.morphologyEx(pal01, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    edge_neigh = cv2.dilate(edge, k_out, iterations=1)
    frx = (frx & edge_neigh).astype(np.uint8)

    # prevent spill deep inside palpebral
    inner_shrink = cv2.erode(pal01, k_in, iterations=1)
    frx[inner_shrink == 1] = 0

    frx = clean(frx, 3)
    return frx




def overlay_color(img, mask01, color_bgr):
    out = img.copy()
    if mask01 is None or mask01.max() == 0:
        return out
    color_layer = np.zeros_like(img, dtype=np.uint8); color_layer[:] = color_bgr
    sel = mask01.astype(bool)
    blended = cv2.addWeighted(img[sel], 1.0 - ALPHA, color_layer[sel], ALPHA, 0)
    out[sel] = blended
    cts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cts:
        cv2.drawContours(out, cts, -1, (0,0,0), 2)
        cv2.drawContours(out, cts, -1, (0,140,255), 1)
    return out




def crop_from_mask(img, mask01, pad=PAD):
    ys, xs = np.where(mask01==1)
    if xs.size == 0: return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = img.shape[:2]
    x0 = max(0, x0-pad); y0 = max(0, y0-pad)
    x1 = min(w, x1+pad+1); y1 = min(h, y1+pad+1)
    return img[y0:y1, x0:x1]
def save_combined_mask(out_dir, base, pal01, frx01):
    """
    Build a 3-class mask: 0=bg, 1=palpebral, 2=fornix.
    Works whether pal/frx are 0/1 or 0/255.
    """
    # normalize to 0/1
    p = (pal01.astype(np.uint8) > 0).astype(np.uint8)
    f = (frx01.astype(np.uint8) > 0).astype(np.uint8)

    comb = np.zeros_like(p, dtype=np.uint8)
    comb[p == 1] = 1
    comb[f == 1] = 2

    out_path = os.path.join(out_dir, f"{base}_combined_mask.png")
    cv2.imwrite(out_path, comb)

    # quick sanity print
    unique, counts = np.unique(comb, return_counts=True)
    print("combined classes:", dict(zip(unique.tolist(), counts.tolist())))
    return out_path
def save_combined_mask_vis(out_dir, base, comb):
    # 0=black, 1=orange, 2=cyan (BGR)
    h, w = comb.shape
    vis = np.zeros((h, w, 3), np.uint8)
    vis[comb == 1] = (0, 85, 255)    # palpebral
    vis[comb == 2] = (255, 255, 0)   # fornix
    path = os.path.join(out_dir, f"{base}_combined_mask_vis.png")
    cv2.imwrite(path, vis)
    return path

def remove_eyelash_spikes_strong(pal01):
    m = (pal01 > 0).astype(np.uint8)
    h, w = m.shape
    y_cut = int(0.72 * h)
    top = m[:y_cut].copy()
    bot = m[y_cut:].copy()

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(35, h // 14)))
    bot_open = cv2.morphologyEx(bot, cv2.MORPH_OPEN, vert_kernel)

    m2 = np.vstack([top, bot_open])

    cleaned = np.zeros_like(m2)
    cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        aspect = hh / max(1, ww)
        mostly_bottom = (y + 0.5*hh) >= y_cut
        is_spike = mostly_bottom and (aspect > 2.5) and (area < 12000)
        if not is_spike:
            cv2.drawContours(cleaned, [c], -1, 1, thickness=-1)

    cleaned[-5:, :] = 0
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return cleaned


def suppress_bottom_lashes(img_bgr, pal01,
                           bottom_ratio=0.35,      # last 35% of image height
                           dark_q=25,              # percentile for "dark" pixels
                           vert_kernel_h_frac=1/14,# vertical kernel height ~ H/14
                           dilate_px=3):
    """
    Build a 'lash mask' from the image (dark, vertical strands near bottom),
    then subtract it from the palpebral mask.
    """
    pal01 = (pal01 > 0).astype(np.uint8)
    h, w = pal01.shape

    # bottom band
    y0 = int((1.0 - bottom_ratio) * h)
    band = np.zeros_like(pal01, np.uint8)
    band[y0:, :] = 1

    # dark pixels in the band
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = int(np.percentile(gray[y0:, :], dark_q))
    dark = (gray <= thr).astype(np.uint8) & band

    # emphasize vertical, strand-like structures
    k_h = max(25, int(h * vert_kernel_h_frac))     # tall vertical kernel
    vert_open = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h)))

    # grow a bit to catch halos
    lash = cv2.dilate(vert_open, np.ones((dilate_px, dilate_px), np.uint8), 1)

    # only remove where pal exists
    pal_clean = pal01.copy()
    pal_clean[lash == 1] = 0

    # small smooth to restore edges
    pal_clean = cv2.morphologyEx(pal_clean, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return pal_clean, lash

def build_lash_mask_color_oriented(img_bgr, pal01,
                                   bottom_ratio=0.40,   # last 40% of image height
                                   dark_q=30,           # "dark" percentile for V/gray
                                   sat_thr=45,          # low-saturation cutoff (HSV)
                                   a_margin=2,          # how much below median a* = not-red
                                   vert_kernel_h_frac=1/12,   # vertical structure kernel
                                   grow_px=3):
    """
    Make a lash mask using COLOR + ORIENTATION:
      - COLOR: (dark) OR (low-saturation AND not-red-in-Lab)
      - ORIENTATION: vertical opening to keep strand-like shapes
      - LOCATION: only in bottom band where lashes live
    Returns: lash_mask (uint8 0/1)
    """
    pal01 = (pal01 > 0).astype(np.uint8)
    h, w = pal01.shape

    # bottom band where lashes live
    y0 = int((1.0 - bottom_ratio) * h)
    band = np.zeros_like(pal01); band[y0:, :] = 1

    # HSV & Lab
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # thresholds (robust to lighting via percentiles/medians)
    v_thr = int(np.percentile(V[y0:, :], dark_q))        # dark
    a_med = np.median(a)

    is_dark      = (V <= v_thr).astype(np.uint8)
    is_low_sat   = (S <= sat_thr).astype(np.uint8)
    is_not_red   = (a <= (a_med - a_margin)).astype(np.uint8)

    # COLOR rule: dark OR (low-sat AND not-red)
    color_lash = (is_dark | (is_low_sat & is_not_red)).astype(np.uint8) & band

    # Emphasize vertical strands
    k_h = max(28, int(h * vert_kernel_h_frac))
    vert_open = cv2.morphologyEx(color_lash, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h)))

    # Grow a little to capture halos/blur
    lash = cv2.dilate(vert_open, np.ones((grow_px, grow_px), np.uint8), 1)

    # Only subtract where pal would otherwise exist (avoid nuking background)
    lash &= pal01

    return lash

def trim_below_lower_envelope(pal01, offset_px=6, smooth_w=81):
    """
    Column-wise lower-envelope trim.
    For each column x, find the *lowest* palpebral pixel (max y),
    smooth that curve, then zero out any palpebral pixels below
    (curve + offset_px).  Kills lash 'teeth' cleanly.
    """
    m = (pal01 > 0).astype(np.uint8)
    h, w = m.shape

    # lowest palpebral y per column (or -1 if none)
    lower = np.full(w, -1, dtype=np.int32)
    ys = np.argmax(m[::-1, :], axis=0)    # from bottom
    has = (m.sum(axis=0) > 0)
    lower[has] = h - 1 - ys[has]          # convert back to image coords

    # fill gaps by nearest interpolation
    if not has.all():
        xs = np.arange(w)
        lower_valid = lower[has]
        if lower_valid.size > 0:
            lower = np.interp(xs, xs[has], lower_valid).astype(np.int32)
        else:
            return m  # nothing to trim

    # smooth with box filter (odd width)
    smooth_w = max(3, smooth_w | 1)
    k = np.ones(smooth_w, dtype=np.float32) / smooth_w
    lower_s = np.convolve(lower.astype(np.float32), k, mode="same")

    # build threshold map and trim
    yy = np.arange(h)[:, None]  # (h,1)
    thr = (lower_s + offset_px)[None, :]  # (1,w)
    keep = (m == 1) & (yy <= thr)
    return keep.astype(np.uint8)
def trim_with_color_floor(img_bgr, pal01, offset_px=4, smooth_w=81):
    """
    Build a per-column 'floor' from red-ish pixels (Lab a* high) inside palpebral,
    smooth it, then remove any palpebral pixels BELOW that floor + offset.
    This ignores dark/low-sat eyelashes automatically.
    """
    pal = (pal01 > 0).astype(np.uint8)
    h, w = pal.shape

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a_med = np.median(a)
    # red-ish mask (tweak +2/+3 if needed)
    redish = ((a >= a_med + 2) & (L >= 25)).astype(np.uint8)

    # only consider red-ish inside palpebral
    cand = (redish & pal).astype(np.uint8)

    # find lowest red-ish pixel per column (ignores dark lashes)
    floor = np.full(w, -1, np.int32)
    # search from bottom; argmax on reversed rows
    ys = np.argmax(cand[::-1, :], axis=0)
    has = (cand.sum(axis=0) > 0)
    floor[has] = h - 1 - ys[has]

    # if some columns missing, interpolate from neighbors
    if not has.all() and has.any():
        xs = np.arange(w)
        floor = np.interp(xs, xs[has], floor[has]).astype(np.int32)
    elif not has.any():
        # fallback: use pal’s lower envelope so we still trim something
        ys2 = np.argmax(pal[::-1, :], axis=0)
        has2 = (pal.sum(axis=0) > 0)
        if not has2.any():
            return pal  # nothing to do
        floor = (h - 1 - ys2).astype(np.int32)

    # smooth the floor so spikes don’t survive
    smooth_w = max(3, smooth_w | 1)  # odd
    k = np.ones(smooth_w, np.float32) / smooth_w
    floor_s = np.convolve(floor.astype(np.float32), k, mode="same")

    # build threshold and trim
    yy = np.arange(h)[:, None]
    thr = (floor_s + offset_px)[None, :]
    keep = (pal == 1) & (yy <= thr)
    return keep.astype(np.uint8)
def remove_thin_strands(pal01, min_width_px=6, bottom_ratio=0.45):
    """
    Kill hair-like 'teeth' inside the palpebral mask purely by thinness.
    - pal01: 0/1 mask for palpebral conjunctiva
    - min_width_px: anything thinner than this is removed
    - bottom_ratio: only act in the bottom X% of the mask's bbox
    """
    m = (pal01 > 0).astype(np.uint8)
    if m.max() == 0:
        return m

    # restrict to bottom of the palpebral bounding box
    ys, xs = np.where(m == 1)
    y0, y1 = ys.min(), ys.max()
    cutoff = int(y0 + (y1 - y0) * (1.0 - bottom_ratio))
    band = np.zeros_like(m); band[cutoff:y1+1, :] = 1
    m_band = (m & band).astype(np.uint8)

    # distance transform → local half-thickness (in pixels)
    dist = cv2.distanceTransform(m_band, cv2.DIST_L2, 5)

    # keep only sufficiently thick areas in the band
    thick_ok = (dist >= (min_width_px / 2.0)).astype(np.uint8)

    # merge: keep all pixels outside band, and only thick pixels inside band
    out = np.zeros_like(m)
    out[m == 1] = 1
    out[band == 1] = thick_ok[band == 1]

    # quick smooth to restore a clean edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out
def main():
    if len(sys.argv) != 3:
        print("Usage: pseudomask_conj.py <image_path> <out_dir>")
        sys.exit(1)

    img_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise FileNotFoundError(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # Country-aware tuning from filename prefix (India_... or Italy_...)
    country = base.split("_", 1)[0].lower()
    if country == "italy":
        # looser redness; thicker fornix ring; slightly deeper lower-bias
        a_delta = 4
        lower_bias_frac = 0.28
        fornix_w = 22
        min_pix = 6000
    else:
        a_delta = 6
        lower_bias_frac = 1/3
        fornix_w = 18
        min_pix = 5000

    # --- Preprocess to reduce domain gap (CLAHE + WB) ---
    img0_proc = preprocess_img(img0)

    # --- Work on a padded image to avoid edge cutoffs ---
    img = pad_img(img0_proc)

    # --- Palpebral from redness (with country-specific knobs) ---
    pal_pad = palpebral_from_redness(img, a_delta=a_delta, lower_bias_frac=lower_bias_frac)
    pal_pad = clean((pal_pad > 0).astype(np.uint8), k=5)

    # --- Fornix ring from palpebral (country-specific ring width) ---
    frx_pad = fornix_from_palpebral(img, pal_pad, width_px=fornix_w, min_pix=min_pix, prefer_bright_lowred=True)

    # --- Unpad back to original size ---
    pal = unpad_mask(pal_pad).astype(np.uint8)
    frx = unpad_mask(frx_pad).astype(np.uint8)

    # -------- Eyelash cleanup ON (helps Italy) --------
    pal = remove_thin_strands(pal, min_width_px=6, bottom_ratio=0.45)
    pal = trim_with_color_floor(img0_proc, pal, offset_px=4, smooth_w=81)
    # pal = remove_eyelash_spikes_strong(pal)  # optional alternative
    # -----------------------------------------------

    # --- Ensure masks are strictly 0/1 ---
    pal01 = (pal > 0).astype(np.uint8)
    frx01 = (frx > 0).astype(np.uint8)

    # --- Save CLEAN combined binary mask (for evaluation) ---
    comb_bin01 = ((pal01 == 1) | (frx01 == 1)).astype(np.uint8)
    comb_bin255 = (comb_bin01 * 255).astype(np.uint8)
    out_comb_mask = os.path.join(out_dir, f"{base}_combined_mask.png")
    cv2.imwrite(out_comb_mask, comb_bin255)

    # --- Save class-coded visualization (0=bg, 1=pal, 2=frx) ---
    comb_cls = np.zeros_like(pal01, dtype=np.uint8)
    comb_cls[pal01 == 1] = 1
    comb_cls[frx01 == 1] = 2
    save_combined_mask_vis(out_dir, base, comb_cls)

    # --- Photo overlay (use original img0 for nicer colors) ---
    combo = overlay_color(overlay_color(img0, pal01, COLOR_PAL), frx01, COLOR_FRX)
    cv2.imwrite(os.path.join(out_dir, f"{base}_combined_overlay.jpg"), combo)

    print("Saved:",
          out_comb_mask, "and",
          os.path.join(out_dir, f"{base}_combined_mask_vis.png"), "and",
          os.path.join(out_dir, f"{base}_combined_overlay.jpg"))








if __name__ == "__main__":
    main()


def combined_overlay(img, pal01, frx01):
    tmp = overlay_color(img, pal01, COLOR_PAL)
    tmp = overlay_color(tmp, frx01, COLOR_FRX)
    return tmp
