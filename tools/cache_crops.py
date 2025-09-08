#!/usr/bin/env python3
import os, csv, sys, json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

# Make repo code importable
sys.path.insert(0, "/workspaces/ishi")

# Use your existing wrapper that calls scripts/pseudomask_conj.py
from api.psuedomask import run_pseudomask_on_bytes  # type: ignore

# ---------- label helpers ----------
def to_bool_label(s: str) -> Optional[bool]:
    s = str(s).strip().lower()
    if s in ("1","anemic","true","yes"): return True
    if s in ("0","not anemic","false","no"): return False
    return None

def read_labels_csv(p: Path) -> Dict[str, str]:
    """
    labels.csv has rows like:
      filename,label
      India_66_20200224_114700_palpebral.png,1
      India_66_20200224_114700_forniceal_palpebral.png,1
    We collapse to STEM -> label, preferring *_palpebral.png over *_forniceal_palpebral.png.
    """
    prefer = ["_palpebral.png", "_forniceal_palpebral.png"]
    tmp = {}
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fname = row["filename"]
            lab = str(row["label"]).strip()
            stem = fname
            for suff in prefer:
                if stem.endswith(suff):
                    stem = stem[:-len(suff)]
                    break
            tmp.setdefault(stem, {})[fname] = lab

    chosen = {}
    for stem, opts in tmp.items():
        pick = None
        for suff in prefer:
            for k in opts.keys():
                if k.endswith(suff):
                    pick = k; break
            if pick: break
        if pick is None:
            pick = next(iter(opts.keys()))
        chosen[stem] = opts[pick]
    return chosen

def stem_from_raw_path(p: Path) -> Optional[str]:
    """
    RAW JPG path -> STEM used in labels.csv
    India/<id>/<YYYYMMDD_HHMMSS>.jpg      => India_{id}_{timestamp}
    Italy/<id>/(<id>.jpg OR T_*.jpg)      => Italy_{id}_{id:03d} OR Italy_{id}_{T_*}
    """
    if p.suffix.lower() != ".jpg": return None
    try:
        country = p.parts[-3]
        subj = p.parts[-2]
        base = p.stem
    except Exception:
        return None
    if country == "India":
        return f"India_{subj}_{base}"
    if country == "Italy":
        if base.startswith("T_"):
            return f"Italy_{subj}_{base}"
        if base.isdigit():
            try:
                if base == subj:
                    return f"Italy_{subj}_{int(base):03d}"
            except:
                pass
    return None

# ---------- image ops ----------
def bgr_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None: raise RuntimeError("cv2.imdecode failed")
    return im

def read_mask_u8(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: raise RuntimeError(f"Failed to read mask {path}")
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def apply_mask_and_crop(bgr: np.ndarray, mask: np.ndarray, pad: int = 6, out_size: int = 256) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(bgr, (out_size, out_size))
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = bgr.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    crop = bgr[y1:y2+1, x1:x2+1]
    return cv2.resize(crop, (out_size, out_size))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia")
    ap.add_argument("--labels-csv", default="/workspaces/ishi/datasets/anemia/processed/eyes_defy_anemia/labels.csv")
    ap.add_argument("--train-list", default="/workspaces/ishi/splits/train.txt")
    ap.add_argument("--val-list",   default="/workspaces/ishi/splits/val.txt")
    ap.add_argument("--out-root",   default="/workspaces/ishi/cached_crops")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    labels = read_labels_csv(Path(args.labels_csv))
    out_root = Path(args.out_root)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    def load_list(p: str) -> list[Path]:
        lst = []
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                # Allow either absolute or relative to data-root
                q = Path(s)
                if not q.is_absolute():
                    q = Path(args.data_root) / s
                lst.append(q)
        return lst

    train_files = load_list(args.train_list)
    val_files   = load_list(args.val_list)

    def process_split(files, split):
        manifest = []
        out_dir = out_root / split
        for i, jpg in enumerate(files, 1):
            if not jpg.exists() or jpg.suffix.lower() != ".jpg":
                continue
            stem = stem_from_raw_path(jpg)
            if not stem or stem not in labels:
                continue
            label_str = labels[stem]
            label = to_bool_label(label_str)
            if label is None:
                continue
            try:
                b = jpg.read_bytes()
                pm = run_pseudomask_on_bytes(b, out_dir=str(out_dir / "pm"), prefix=f"{stem}")
                mask_path = pm.get("mask_path")
                if not mask_path:
                    print(f"[WARN] No mask for {jpg}")
                    continue
                bgr = bgr_from_bytes(b)
                mask = read_mask_u8(mask_path)
                crop = apply_mask_and_crop(bgr, mask, pad=6, out_size=args.size)
                out_img = out_dir / f"{stem}.png"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), crop)
                manifest.append({"image": str(out_img), "label": int(label)})
            except Exception as e:
                print(f"[ERR] {jpg}: {e}")
                continue
        mf_path = out_dir / "manifest.csv"
        with open(mf_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image","label"])
            w.writeheader()
            w.writerows(manifest)
        print(f"[{split}] saved {len(manifest)} crops -> {mf_path}")
        return mf_path

    tr_csv = process_split(train_files, "train")
    va_csv = process_split(val_files, "val")

    meta = {
        "train_manifest": str(tr_csv),
        "val_manifest": str(va_csv),
        "size": args.size
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
