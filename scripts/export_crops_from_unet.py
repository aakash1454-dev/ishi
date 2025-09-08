# scripts/export_crops_from_unet.py
import sys
from pathlib import Path
import argparse
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.crop.train_unet import UNetSmall, EyelidSegDataset, DEVICE
from api.utils.mask_ops import refine_mask_from_prob, crop_from_mask_with_pad

def to_uint8_rgb(t3):
    return (t3.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# --- put this near the top of export_crops_from_unet.py ---
import re
from pathlib import Path

def _norm_key(s: str) -> str:
    """Loosely normalize by lowercasing, swapping separators to '_', and collapsing runs."""
    s = s.replace("\\", "/").lower()
    s = s.strip()
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s

def path_to_csvkey(p: Path, dataset_root: Path) -> str:
    """
    Convert a disk path Country/Subject/file.png into the CSV style:
    Country_Subject_file.png  (then normalized)
    """
    rel = p.relative_to(dataset_root)  # e.g. India/39/20200213_...png
    parts = rel.parts                   # ("India","39","20200213_...png")
    if len(parts) < 3:
        # fallback: just normalize the filename
        return _norm_key(rel.name)
    country, subject, filename = parts[0], parts[1], parts[-1]
    flat = f"{country}_{subject}_{filename}"
    return _norm_key(flat)

def load_labels_csv(csv_path: Path) -> dict:
    """
    Returns dict: {normalized_csv_filename: int_label}
    Where keys are normalized like _norm_key("India_39_...png")
    """
    import csv
    label_map = {}
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    # skip header if present
    start = 1 if rows and rows[0] and rows[0][0].lower() == "filename" else 0
    for row in rows[start:]:
        if len(row) < 2: 
            continue
        fname, lab = row[0], row[1]
        key = _norm_key(fname)
        try:
            y = int(lab)
        except Exception:
            continue
        label_map[key] = y
    return label_map

def find_label_for_path(path: Path, label_map: dict, dataset_root: Path):
    """
    Build the CSV-style key from a disk path and look it up.
    """
    key = path_to_csvkey(path, dataset_root)
    return label_map.get(key, None)


def _candidate_keys_for_path(p: Path, dataset_root: Path = None):
    p = p.as_posix()
    parts = p.split("/")
    keys = set()
    # full
    keys.add(_norm_key(p))
    # basename & stem
    keys.add(_norm_key(Path(p).name))
    keys.add(_norm_key(Path(p).stem))
    # last-2 / last-3 components
    if len(parts) >= 2:
        keys.add(_norm_key("/".join(parts[-2:])))
    if len(parts) >= 3:
        keys.add(_norm_key("/".join(parts[-3:])))
    # relative to dataset_root if provided
    if dataset_root is not None:
        try:
            rel = Path(p).resolve().relative_to(dataset_root.resolve())
            keys.add(_norm_key(rel.as_posix()))
        except Exception:
            pass
    return keys

def load_labels_csv(csv_path: Path):
    """
    Accepts CSV with (filename,label) or (path,label) (header or no header).
    Returns dict mapping many normalized keys -> {0,1}.
    """
    mapping = {}
    def add_row(path_str, y):
        y = int(y)
        # generate many keys
        P = Path(path_str)
        for k in _candidate_keys_for_path(P):
            mapping[k] = y

    with open(csv_path, "r", newline="") as f:
        # try DictReader first
        sniffer = csv.Sniffer()
        sample = f.read(1024)
        f.seek(0)
        has_header = False
        try:
            has_header = sniffer.has_header(sample)
        except Exception:
            pass

        if has_header:
            reader = csv.DictReader(f)
            # try common column names
            cols = [c.lower() for c in reader.fieldnames or []]
            # filename/ path column
            name_col = None
            for cand in ("filename","file","image","path","filepath"):
                if cand in cols:
                    name_col = reader.fieldnames[cols.index(cand)]
                    break
            # label column
            label_col = None
            for cand in ("label","class","target","y"):
                if cand in cols:
                    label_col = reader.fieldnames[cols.index(cand)]
                    break
            if name_col is None or label_col is None:
                raise ValueError(f"CSV needs filename/path and label columns. Found: {reader.fieldnames}")
            for row in reader:
                add_row(row[name_col], row[label_col])
        else:
            # simple: filename,label
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    add_row(parts[0], parts[1])
    return mapping

def find_label_for_path(ipath: Path, label_map: dict, dataset_root: Path = None):
    for k in _candidate_keys_for_path(ipath, dataset_root):
        k2 = _norm_key(k)
        if k2 in label_map:
            return int(label_map[k2])
    return None

def guess_image_path(ds, idx):
    """
    Pull the original image path for index from EyelidSegDataset.
    """
    if hasattr(ds, "pairs"):
        rec = ds.pairs[idx]
        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
            return Path(rec[0])
    # fallback attempts
    for attr in ("_pairs","items","_items","paths","_paths","images","_images"):
        if hasattr(ds, attr):
            try:
                rec = getattr(ds, attr)[idx]
                if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                    return Path(rec[0])
                if isinstance(rec, (str, Path)):
                    return Path(rec)
            except Exception:
                pass
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="Country dirs (India, Italy, ...)")
    parser.add_argument("--ckpt", required=True, help="UNet checkpoint")
    parser.add_argument("--csv", required=True, help="CSV with filename,label")
    parser.add_argument("--out", required=True, help="Output dir for crops")
    parser.add_argument("--pad", type=float, default=0.12, help="Padding ratio around bbox")
    parser.add_argument("--min_side", type=int, default=80, help="Minimum side of crop")
    # near argparse setup
    parser.add_argument(
        "--restrict-to-csv", action="store_true",
        help="Export only images that have a matching entry in the CSV"
)

    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # For relative key generation, pick the common parent of provided roots (their parent dir)
    # e.g., .../dataset_anemia
    dataset_root = roots[0].parent if roots else None

    # labels
    label_map = load_labels_csv(Path(args.csv))

    # model
    model = UNetSmall().to(DEVICE)
    ckpt = torch.load(Path(args.ckpt), map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # dataset
    ds = EyelidSegDataset(roots)
    print(f"Found {len(ds)} pairs.")

    counts = {"anemic": 0, "nonanemic": 0, "unknown": 0, "skipped": 0}

    for idx in tqdm(range(len(ds)), desc="Exporting crops"):
        try:
            img_t, _ = ds[idx]
            img_np = to_uint8_rgb(img_t)

            with torch.no_grad():
                prob = torch.sigmoid(model(img_t.unsqueeze(0).to(DEVICE)))[0,0].cpu().numpy()

            mask = refine_mask_from_prob(prob)
            crop = crop_from_mask_with_pad(img_np, mask, pad=args.pad, min_side=args.min_side)
            if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                counts["skipped"] += 1
                continue

            # --- label lookup + optional restriction ---
            ipath = guess_image_path(ds, idx)
            y = find_label_for_path(ipath, label_map, dataset_root) if ipath is not None else None

            # If user asked to export only labeled items, skip unlabeled
            if args.restrict_to_csv and y is None:
                counts["skipped"] += 1
                continue

            label_str = "anemic" if y == 1 else ("nonanemic" if y == 0 else "unknown")

            # --- write output ---
            out_subdir = (out_dir / label_str); out_subdir.mkdir(parents=True, exist_ok=True)
            out_path = out_subdir / f"{idx:05d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            counts[label_str] += 1


        except Exception:
            counts["skipped"] += 1
            continue

    print("Done.")
    print(f"Saved crops -> {out_dir}")
    print(f"Counts: anemic={counts['anemic']}  nonanemic={counts['nonanemic']}  unknown={counts['unknown']}  skipped={counts['skipped']}")
