#!/usr/bin/env python3
import os, csv, sys, json, argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

# Ensure repo on path
sys.path.insert(0, "/workspaces/ishi")
from api.psuedomask import run_pseudomask_on_bytes, DEFAULT_API_OUT  # type: ignore
from api.anemia_classifier import analyze_with_mask  # type: ignore

# ---------------- Helpers ----------------
def to_bool_label(s: str) -> Optional[bool]:
    s = s.strip().lower()
    if s in ("1", "anemic", "true", "yes"): return True
    if s in ("0", "not anemic", "false", "no"): return False
    return None

def read_labels(labels_csv: Path) -> Dict[str, str]:
    """
    Read labels.csv with header 'filename,label' and return dict: STEM -> label
    STEM is filename with region suffix stripped:
      *_palpebral.png or *_forniceal_palpebral.png removed
    If both exist, prefer palpebral.
    """
    prefer_order = ["_palpebral.png", "_forniceal_palpebral.png"]
    by_stem: Dict[str, Dict[str, str]] = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("file") or ""
            lab = str(row.get("label", "")).strip()
            if not fname: continue
            stem = fname
            for suff in prefer_order:
                if stem.endswith(suff):
                    stem = stem[:-len(suff)]
                    break
            by_stem.setdefault(stem, {})[fname] = lab

    chosen: Dict[str, str] = {}
    for stem, options in by_stem.items():
        pick = None
        for suff in prefer_order:
            for k in options.keys():
                if k.endswith(suff):
                    pick = k; break
            if pick: break
        if pick is None:
            pick = next(iter(options.keys()))
        chosen[stem] = options[pick]
    return chosen

def stem_from_raw_path(p: Path) -> Optional[str]:
    """
    Build label STEM from RAW jpg path.
    - India: India/<id>/<YYYYMMDD_HHMMSS>.jpg  -> India_{id}_{timestamp}
    - Italy (variant A): Italy/<id>/<id>.jpg   -> Italy_{id}_{id:03d}
    - Italy (variant B): Italy/<id>/T_*.jpg    -> Italy_{id}_{T_*}
    Returns None if pattern not recognized.
    """
    if p.suffix.lower() != ".jpg": return None
    try:
        country = p.parts[-3]   # .../<Country>/<id>/<file>.jpg
        subj_id = p.parts[-2]
        base = p.stem
    except Exception:
        return None

    if country == "India":
        # base like 20200224_114700
        return f"India_{subj_id}_{base}"

    if country == "Italy":
        if base.startswith("T_"):
            # T_* form
            return f"Italy_{subj_id}_{base}"
        # digits equal to folder id → pad to 3 digits
        if base.isdigit():
            try:
                n = int(subj_id)
                if base == subj_id:
                    return f"Italy_{subj_id}_{int(base):03d}"
            except ValueError:
                pass
    return None

def bgr_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("cv2.imdecode failed")
    return img

def read_mask_u8(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: raise RuntimeError(f"Failed to read mask: {path}")
    _, m2 = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m2

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia",
                    help="Root folder containing India/ and Italy/ RAW JPGs")
    ap.add_argument("--labels", default="/workspaces/ishi/datasets/anemia/processed/eyes_defy_anemia/labels.csv",
                    help="Path to labels.csv with columns: filename,label")
    ap.add_argument("--outdir", default="/workspaces/ishi/out_eval", help="Output directory")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    labels_csv = Path(args.labels)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "raw_results.csv"
    mistakes_csv = outdir / "raw_mistakes.csv"

    gt = read_labels(labels_csv)

    rows = []
    tp = tn = fp = fn = total = 0
    scanned = 0
    for country in ["India", "Italy"]:
        cdir = data_root / country
        if not cdir.exists(): continue
        for root, _, files in os.walk(cdir):
            for fname in files:
                if not fname.lower().endswith(".jpg"): continue
                scanned += 1
                fpath = Path(root) / fname
                stem = stem_from_raw_path(fpath)
                if not stem:  # unrecognized pattern
                    continue
                if stem not in gt:
                    # not in labels; skip from eval to keep in-sync with CSV
                    continue

                true_is_anemic = to_bool_label(gt[stem])

                # run pipeline (mask+classify)
                try:
                    img_bytes = fpath.read_bytes()
                    pm = run_pseudomask_on_bytes(img_bytes, out_dir=str(outdir / "pmask_runs"), prefix="eval")
                    mask_path = pm.get("mask_path")
                    if not mask_path or not os.path.exists(mask_path):
                        print(f"[ERR] No mask for {fpath}")
                        continue
                    bgr = bgr_from_bytes(img_bytes)
                    mask = read_mask_u8(mask_path)
                    out = analyze_with_mask(bgr, mask, include_images_b64=False)
                    pred_label = out.get("label", "Unknown")
                    prob = float(out.get("probability", 0.0))
                    pred_is_anemic = pred_label.lower().startswith("anemic")
                except Exception as e:
                    print(f"[ERR] Failed on {fpath}: {e}")
                    continue

                cover = float((mask > 0).sum()) / float(mask.size)

                # stats
                if true_is_anemic is not None:
                    total += 1
                    if pred_is_anemic and true_is_anemic: tp += 1
                    elif (not pred_is_anemic) and (not true_is_anemic): tn += 1
                    elif (not pred_is_anemic) and true_is_anemic: fn += 1
                    elif pred_is_anemic and (not true_is_anemic): fp += 1

                rows.append({
                    "file": str(fpath),
                    "stem": stem,
                    "pred_label": pred_label,
                    "pred_prob": prob,
                    "true_label": gt[stem],
                    "mask_cover": cover,
                })

    # write all results
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","stem","pred_label","pred_prob","true_label","mask_cover"])
        writer.writeheader()
        writer.writerows(rows)

    # write mistakes only
    mistakes = []
    for r in rows:
        t = to_bool_label(r["true_label"])
        p = r["pred_label"].lower().startswith("anemic")
        if t is None: continue
        if (p and not t) or ((not p) and t):
            mistakes.append(r)
    with open(mistakes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","stem","pred_label","pred_prob","true_label","mask_cover"])
        writer.writeheader()
        writer.writerows(mistakes)

    summary = {
        "scanned_jpgs": scanned,
        "evaluated": total,
        "correct": tp + tn,
        "incorrect": fp + fn,
        "accuracy": round(((tp + tn) / total), 4) if total else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "results_csv": str(results_csv),
        "mistakes_csv": str(mistakes_csv),
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
