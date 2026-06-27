#!/usr/bin/env python3
"""
Build the subject-aware conjunctiva crop set + manifest.csv for the model work.

One uniform path for all 216 subjects, at the canonical 800x1067 (EXIF-corrected)
frame: original .jpg (EXIF-corrected) cropped to the expert hand-mask bounding box.
The Italy(1-31) early batch is included on this same path because its white-bg masks
were normalized to canonical RGBA first (scripts/normalize_masks.py); those subjects
are tagged source="hand_early" for slice/audit.

One crop per subject (fullest available conjunctiva view), tight bbox + small pad.
Labels (anemic/nonanemic) + Hb come from the Excel-derived labels.csv, joined on
(country, subject_id).

Run (after scripts/normalize_masks.py):
  uv run python scripts/build_crops_manifest.py
"""
import argparse
import csv
import pathlib

import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

RAW = pathlib.Path("datasets/anemia/raw/eyes_defy_anemia/dataset_anemia")
LABELS_CSV = pathlib.Path("datasets/anemia/processed/eyes_defy_anemia/labels.csv")
FRAME = (800, 1067)
MASK_PREFERENCE = ("forniceal_palpebral", "palpebral")


def best_mask(folder):
    for kind in MASK_PREFERENCE:
        for m in folder.glob(f"*_{kind}.png"):
            return m
    return None


def load_labels():
    """(country, subject_id) -> (hb, label) with label in {anemic, nonanemic}."""
    out = {}
    for r in csv.DictReader(open(LABELS_CSV)):
        lf = r["label_final"].strip()
        if lf == "":
            continue
        lab = "anemic" if float(lf) == 1.0 else "nonanemic"
        out[(r["country"], r["subject_id"])] = (r["hb"].strip(), lab)
    return out


def crop_to_mask(img_pil, mask_arr, pad_frac=0.06):
    """Crop img (PIL, FRAME) to the bounding box of mask (bool HxW) + padding."""
    ys, xs = np.where(mask_arr)
    if xs.size == 0:
        return None
    W, H = img_pil.size
    pad = int(pad_frac * max(W, H))
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, W)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, H)
    return img_pil.crop((x0, y0, x1, y1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-crops", default="data/crops")
    ap.add_argument("--out-manifest", default="data/manifest.csv")
    args = ap.parse_args()

    labels = load_labels()
    out_dir = pathlib.Path(args.out_crops); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    # ---- 186 good subjects: crop from expert hand-mask ----
    for country in ("India", "Italy"):
        for d in (RAW / country).iterdir():
            if not d.is_dir():
                continue
            jpg = list(d.glob("*.jpg")); m = best_mask(d)
            if not jpg or m is None:
                continue
            try:
                marr = np.array(Image.open(m).convert("RGBA").resize(FRAME, Image.NEAREST))[..., 3] > 0
            except Exception:
                continue
            cov = marr.mean()
            if not (0.005 < cov < 0.90):    # skip degenerate/empty (the 31 handled below)
                continue
            key = (country, d.name)
            if key not in labels:
                skipped.append(f"{country}_{d.name}(no-label)"); continue
            img = ImageOps.exif_transpose(Image.open(jpg[0])).convert("RGB").resize(FRAME)
            crop = crop_to_mask(img, marr)
            if crop is None:
                skipped.append(f"{country}_{d.name}(empty-bbox)"); continue
            name = f"{country}_{d.name}.png"
            crop.save(out_dir / name)
            hb, lab = labels[key]
            # Italy 1-31 is the early acquisition batch (masks normalized from
            # white-bg by scripts/normalize_masks.py) -> tag for slice/audit.
            is_early = country == "Italy" and d.name.isdigit() and 1 <= int(d.name) <= 31
            rows.append([name, country, d.name, hb, lab, "hand_early" if is_early else "hand"])

    rows.sort(key=lambda r: (r[1], int(r[2])))
    with open(args.out_manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "country", "subject_id", "hb", "label", "source"])
        w.writerows(rows)

    import collections
    print(f"crops written: {len(rows)} -> {out_dir}")
    print("by source:", dict(collections.Counter(r[5] for r in rows)))
    print("by label :", dict(collections.Counter(r[4] for r in rows)))
    print("by country:", dict(collections.Counter(r[1] for r in rows)))
    print(f"distinct subjects: {len(set((r[1], r[2]) for r in rows))}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:8]}{'...' if len(skipped)>8 else ''}")
    print(f"manifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()
