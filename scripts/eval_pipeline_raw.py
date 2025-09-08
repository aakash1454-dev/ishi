#!/usr/bin/env python3
import os, sys, csv, glob, json, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T

# Import your existing helpers
# - run_pseudomask_on_bytes returns {"mask_path": ...}
# Adjust import path if your package layout differs.
sys.path.append(str(Path(__file__).resolve().parents[1] / "api"))
from api.psuedomask import run_pseudomask_on_bytes, DEFAULT_API_OUT

# Reuse model builder from training
sys.path.append(str(Path(__file__).resolve().parents[1] / "training/anemia"))
from train import build_model

def load_labels_csv(csv_path: Path):
    """
    Robust loader: auto-detect filename/id and label columns.
    Maps labels to {'anemic','nonanemic'}.
    Matches by filename stem (case-insensitive). Also tries numeric ID (ignores leading zeros).
    """
    if not csv_path or not csv_path.exists():
        return {}

    def pick(cols, candidates):
        cl = [c.lower() for c in cols]
        for cand in candidates:
            if cand in cl:
                return cols[cl.index(cand)]
        # fallback: first that contains any token
        for i,c in enumerate(cl):
            if any(tok in c for tok in ["file", "image", "img", "name", "path", "id"]):
                return cols[i]
        return None

    def norm_label(v):
        s = str(v).strip().lower()
        if s in ("anemic","anemia","pos","positive","1","true","yes","y"):
            return "anemic"
        if s in ("nonanemic","non_anemic","not anemic","normal","healthy","neg","negative","0","false","no","n"):
            return "nonanemic"
        # last resort: treat others as unknown
        return None

    labmap = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        fname_col = pick(cols, ["filename","file","image","img","path","name","id"])
        label_col = pick(cols, ["label","class","anemic","status","gt"])
        if not fname_col or not label_col:
            # Give up gracefully
            return {}

        for r in reader:
            fn_raw = (r.get(fname_col) or "").strip()
            lbl_raw = r.get(label_col)
            lbl = norm_label(lbl_raw)
            if not fn_raw or lbl is None:
                continue

            p = Path(fn_raw)
            stem = p.stem.lower() if p.suffix else fn_raw.strip().lower()

            # store multiple keys for matching:
            keys = {stem}
            # numeric-id variant (ignore leading zeros)
            digits = "".join([ch for ch in stem if ch.isdigit()])
            if digits:
                keys.add(str(int(digits)))  # "00029" -> "29"

            for k in keys:
                labmap[k] = lbl
    return labmap


def infer_tf(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def _read_mask_u8(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask {path}")
    _, m2 = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m2

def _mask_bbox_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    h, w = mask.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    return bbox_area / float(h * w)

def _crop_with_mask_or_skip(bgr: np.ndarray, mask: np.ndarray,
                            min_bbox_ratio=0.15, max_bbox_ratio=0.95,
                            small_side_skip=512, pad=6) -> Image.Image:
    """Guarded crop: skip if bbox too small/large or image already small."""
    h, w = bgr.shape[:2]
    if min(h, w) < small_side_skip or mask is None:
        crop = bgr
    else:
        r = _mask_bbox_ratio(mask)
        if r < min_bbox_ratio or r > max_bbox_ratio:
            crop = bgr
        else:
            ys, xs = np.where(mask > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0)
            x1 = min(x1 + pad, w); y1 = min(y1 + pad, h)
            crop = bgr[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)

def list_images(raw_dir: Path):
    exts = ("*.png","*.jpg","*.jpeg","*.webp","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts:
        files += glob.glob(str(raw_dir / e))
    files = sorted(files)
    return [Path(p) for p in files]

def main():
    ap = argparse.ArgumentParser(description="End-to-end eval: pseudomask -> guarded crop -> CNN classifier")
    ap.add_argument("--raw-dir", required=True, type=str, help="Folder with ~305 RAW images")
    ap.add_argument("--labels-csv", type=str, default="", help="CSV with columns filename,label (optional but recommended)")
    ap.add_argument("--ckpt", type=str, default="runs/anemia_resnet18_v1/best_acc.pth")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=float(os.getenv("ANEMIA_THRESHOLD","0.75")))
    ap.add_argument("--out-csv", type=str, default="runs/eval/eval_pipeline_all.csv")
    ap.add_argument("--backbone", type=str, default="", help="Override backbone if needed; otherwise read from ckpt")
    ap.add_argument("--use-mask", action="store_true", help="Force using mask-guided crop even on small images")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    assert raw_dir.is_dir(), f"RAW dir not found: {raw_dir}"

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone = args.backbone or ckpt.get("backbone","resnet18")
    classes = ckpt.get("classes", ["anemic","nonanemic"])
    model = build_model(num_classes=2, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = infer_tf(args.img_size)

    # Labels
    labmap = load_labels_csv(Path(args.labels_csv)) if args.labels_csv else {}

    # Collect files
    files = list_images(raw_dir)
    if not files:
        print(f"No images found in {raw_dir}")
        sys.exit(1)

    rows = []
    for i, p in enumerate(files, 1):
        # bytes for mask
        with open(p, "rb") as f:
            data = f.read()

        # mask (best-effort; ok if fails)
        mask = None
        try:
            pm = run_pseudomask_on_bytes(data, out_dir=DEFAULT_API_OUT, prefix="pmask")
            mpath = pm.get("mask_path")
            if mpath and os.path.isfile(mpath):
                mask = _read_mask_u8(mpath)
        except Exception as e:
            mask = None  # continue without mask

        # decode original
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] imdecode failed: {p}")
            continue

        # guarded crop (or forced mask if user asked)
        if args.use_mask:
            pil = _crop_with_mask_or_skip(bgr, mask, min_bbox_ratio=0.0, max_bbox_ratio=1.0, small_side_skip=0)
        else:
            pil = _crop_with_mask_or_skip(bgr, mask)  # guarded defaults

        # inference
        x = tf(pil).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx_anemic = classes.index("anemic")
        p_anemic = float(probs[idx_anemic])
        pred = "anemic" if p_anemic >= args.threshold else "nonanemic"

        # true label if provided
        key = p.stem.lower()
        true = labmap.get(key, "")

        rows.append({
            "path": str(p),
            "true": true,
            "pred": pred,
            "p_anemic": p_anemic
        })

        if i % 25 == 0:
            print(f"Processed {i}/{len(files)}")

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","true","pred","p_anemic"])
        w.writeheader()
        w.writerows(rows)
    print("Saved:", args.out_csv)

    # Metrics if labels available
    labeled = [r for r in rows if r["true"] in ("anemic","nonanemic")]
    if labeled:
        y_true = np.array([1 if r["true"]=="anemic" else 0 for r in labeled], int)
        y_pred = np.array([1 if r["pred"]=="anemic" else 0 for r in labeled], int)
        tp = int(((y_pred==1)&(y_true==1)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        acc  = (tp+tn)/len(labeled)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0

        print(f"\nLabeled images: {len(labeled)} / {len(rows)} total")
        print(f"(tp, fp, tn, fn) = ({tp}, {fp}, {tn}, {fn})")
        print(f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  spec={spec:.3f}  f1={f1:.3f}")
    else:
        print("\nNo labels found in CSV; skipped metrics.")
if __name__ == "__main__":
    main()