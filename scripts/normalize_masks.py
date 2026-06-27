#!/usr/bin/env python3
"""
Normalize the Italy 1-31 'white-background' conjunctiva segmentation PNGs into the
SAME format the rest of the dataset uses: RGBA where
  - alpha = mask (255 on the conjunctiva, 0 on background)
  - RGB   = conjunctiva colour on black
This lets one uniform (alpha-based) code path read every mask. Originals are backed
up to data/rescue_italy_1_31/raw_seg_backup/ before any file is overwritten.

Run:
  uv run python scripts/normalize_masks.py
"""
import pathlib
import shutil

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW = pathlib.Path("datasets/anemia/raw/eyes_defy_anemia/dataset_anemia/Italy")
BACKUP = pathlib.Path("data/rescue_italy_1_31/raw_seg_backup")
KINDS = ("forniceal_palpebral", "palpebral", "forniceal")
WHITE = 230  # a pixel is "background" if all channels exceed this


def is_whitebg(arr):
    """RGBA array -> True if it's the degenerate white-bg encoding."""
    a = arr[..., 3]
    white_frac = (arr[..., :3].min(2) > WHITE).mean()
    return a.min() > 250 and white_frac > 0.30  # fully opaque alpha + lots of white


def normalize(path):
    arr = np.array(Image.open(path).convert("RGBA"))
    if not is_whitebg(arr):
        return False
    rgb = arr[..., :3]
    mask = ~(rgb.min(2) > WHITE)              # conjunctiva = non-white
    out = np.zeros_like(arr)
    out[..., :3] = np.where(mask[..., None], rgb, 0)
    out[..., 3] = (mask * 255).astype(np.uint8)
    # back up original (once), then overwrite in canonical format
    rel = path.relative_to(RAW.parent)        # Italy/<n>/<file>.png
    bdst = BACKUP / rel
    bdst.parent.mkdir(parents=True, exist_ok=True)
    if not bdst.exists():
        shutil.copy2(path, bdst)
    Image.fromarray(out, "RGBA").save(path)
    return True


def main():
    checked = changed = 0
    for n in range(1, 32):
        d = RAW / str(n)
        if not d.is_dir():
            continue
        for kind in KINDS:
            for p in d.glob(f"*_{kind}.png"):
                checked += 1
                if normalize(p):
                    changed += 1
    print(f"checked {checked} segmentation files, normalized {changed}")
    print(f"originals backed up under {BACKUP}")


if __name__ == "__main__":
    main()
