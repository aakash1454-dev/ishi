#!/usr/bin/env python3
# scripts/slice_report.py
import os, re, argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List

# ---------- Paths (defaults; override with CLI if you want) ----------
DEF_RESNET = "runs/eval/tta_resnet18_labeled.csv"
DEF_MBNET  = "runs/eval/tta_mobilenetv3_labeled.csv"
DEF_STACK  = "runs/eval/stacked_labeled.csv"
OUT_DIR    = "runs/eval"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def ensure_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize path columns:
      - Accept 'filepath' or 'filename' or 'img' etc.
      - Add 'basename' from whichever is found.
    """
    cand = ["filepath", "path", "img", "image", "file", "filename", "image_path"]
    src = None
    for c in cand:
        if c in df.columns:
            src = c
            break
    if src is None:
        raise RuntimeError("Input needs a file-path-like column (looked for one of: "
                           f"{cand}). Found: {list(df.columns)}")
    df = df.copy()
    df[src] = df[src].astype(str).str.replace("\\\\","/", regex=False)
    df["filepath"] = df[src] if "filepath" not in df.columns else df["filepath"]
    df["basename"] = df["filepath"].apply(lambda p: Path(p).name)
    return df

def pick_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "class", "target", "gt", "y", "is_anemic", "anemic"]:
        if c in df.columns: return c
    raise RuntimeError("Could not find a ground-truth label column in "
                       f"{list(df.columns)}. Expected one of: label/class/target/gt/y/is_anemic/anemic.")

def ensure_pred_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = {"prob_anemic", "pred_label"}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise RuntimeError(f"Prediction file missing required columns: {miss}. "
                           f"Have: {list(df.columns)}")
    df = df.copy()
    df["pred_label"] = df["pred_label"].astype(int)
    return df

def infer_country(basename: str) -> str:
    # Expect names like "India_..." or "Italy_..."
    m = re.match(r"^(India|Italy)[^_]*_", basename)
    if m: return m.group(1)
    return "unknown"

def infer_view(basename: str) -> str:
    b = basename.lower()
    if "forniceal" in b: return "forniceal_palpebral"
    if "palpebral" in b: return "palpebral"
    return "unknown"

def add_slices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country"] = df["basename"].apply(infer_country)
    df["view"]    = df["basename"].apply(infer_view)
    return df

def metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return dict(n=0,tp=0,tn=0,fp=0,fn=0,acc=0,prec=0,rec=0,f1=0)
    y = df["label"].astype(int).values
    p = df["pred_label"].astype(int).values
    tp = int(((y==1)&(p==1)).sum()); tn = int(((y==0)&(p==0)).sum())
    fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    n = len(df)
    acc = (tp+tn)/n if n else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return dict(n=n,tp=tp,tn=tn,fp=fp,fn=fn,acc=acc,prec=prec,rec=rec,f1=f1)

def per_slice(df: pd.DataFrame, slice_cols: List[str]) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(slice_cols, dropna=False):
        if not isinstance(name, tuple): name = (name,)
        row = {col: val for col, val in zip(slice_cols, name)}
        row.update(metrics(g))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(slice_cols).reset_index(drop=True)

def brief_table(df: pd.DataFrame, model_tag: str, slice_cols: List[str]):
    cols = slice_cols + ["n","acc","prec","rec","f1","tp","tn","fp","fn"]
    print(f"\n== {model_tag} slice metrics ==")
    if df.empty:
        print("(no rows)")
    else:
        print(df[cols].to_string(index=False))

# ---------- Loaders ----------
def load_pred_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_paths(df)
    df = ensure_pred_cols(df)

    # Guarantee ground-truth label exists and is 0/1
    if "label" not in df.columns:
        # some labeled CSVs may call it 'label' already; if not, we won’t invent it here
        raise RuntimeError(f"{path} has no 'label' column. "
                           f"Re-run compare/labeling step to create *_labeled.csv.")
    df["label"] = df["label"].astype(int)
    df = add_slices(df)
    return df

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default=DEF_RESNET)
    ap.add_argument("--mbnet",  default=DEF_MBNET)
    ap.add_argument("--stack",  default=DEF_STACK)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    have = {}
    for tag, p in [("ResNet18", args.resnet), ("MobileNetV3", args.mbnet), ("Stacked", args.stack)]:
        if os.path.exists(p):
            try:
                have[tag] = load_pred_csv(p)
            except Exception as e:
                print(f"[WARN] Skipping {tag} at {p}: {e}")
        else:
            print(f"[WARN] Missing file for {tag}: {p}")

    if not have:
        raise SystemExit("No valid prediction CSVs found. Expected labeled CSVs with 'label' + predictions.")

    slice_cols_sets = [
        (["country"],      "by_country"),
        (["view"],         "by_view"),
        (["country","view"], "by_country_view")
    ]

    all_summaries = {}
    for tag, df in have.items():
        for cols, name in slice_cols_sets:
            rep = per_slice(df, cols)
            brief_table(rep, f"{tag} {name}", cols)
            out = os.path.join(args.out_dir, f"slice_{tag.lower()}_{name}.csv")
            rep.to_csv(out, index=False)
            all_summaries[(tag, name)] = out

    print("\nSaved slice CSVs:")
    for (tag, name), p in all_summaries.items():
        print(f"  - {tag:12s} {name:15s} -> {p}")

if __name__ == "__main__":
    main()
