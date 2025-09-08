#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

# -------- Paths (edit if yours differ) --------
LABELS      = "datasets/anemia/processed/eyes_defy_anemia/labels.csv"
RESNET_PRED = "runs/eval/tta_resnet18.csv"
MBNET_PRED  = "runs/eval/tta_mobilenetv3.csv"

OUT_DIR          = "runs/eval"
RESNET_LABELED   = os.path.join(OUT_DIR, "tta_resnet18_labeled.csv")
MBNET_LABELED    = os.path.join(OUT_DIR, "tta_mobilenetv3_labeled.csv")
RESNET_FP        = os.path.join(OUT_DIR, "resnet18_fp.csv")
RESNET_FN        = os.path.join(OUT_DIR, "resnet18_fn.csv")
MBNET_FP         = os.path.join(OUT_DIR, "mobilenetv3_fp.csv")
MBNET_FN         = os.path.join(OUT_DIR, "mobilenetv3_fn.csv")
COMPARE_CSV      = os.path.join(OUT_DIR, "resnet18_vs_mobilenetv3_compare.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Helpers --------
PATH_CANDS  = ["filepath","path","img","image","image_path","file","filename"]
LABEL_CANDS = ["label","anemic","class","target","y","gt","ground_truth","is_anemic","status"]

def _find_col(df, cands, desc):
    for c in cands:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"labels.csv needs a {desc} column. Looked for any of {cands}. "
        f"Found: {list(df.columns)}"
    )

def _norm_label(v):
    if pd.isna(v): return None
    if isinstance(v,(int,float)): return 1 if int(v)>=1 else 0
    s = str(v).strip().lower().replace("-","_")
    pos = {"1","true","yes","y","pos","positive","anemic"}
    neg = {"0","false","no","n","neg","negative","non_anemic","healthy","normal","nonanemic","non_anemic"}
    if s in pos: return 1
    if s in neg: return 0
    try:
        return 1 if int(float(s))>=1 else 0
    except:
        return None

def load_labels(path):
    df = pd.read_csv(path)
    pcol = _find_col(df, PATH_CANDS, "file-path-like")
    # pick/guess label column
    lcol = None
    for c in LABEL_CANDS:
        if c in df.columns:
            lcol = c
            break
    if lcol is None:
        others = [c for c in df.columns if c != pcol]
        if not others:
            raise RuntimeError("Could not infer label column.")
        lcol = min(others, key=lambda c: df[c].nunique())
        print(f"[INFO] Guessed label column: '{lcol}' (unique={df[lcol].nunique()})")
    df = df[[pcol, lcol]].copy()
    df.rename(columns={pcol:"filepath", lcol:"label_raw"}, inplace=True)
    df["label"] = df["label_raw"].map(_norm_label)
    if df["label"].isna().any():
        bad = df[df["label"].isna()].head(8)
        uniq = df["label_raw"].dropna().astype(str).str.lower().unique()[:20]
        raise RuntimeError(
            "Some labels could not be parsed into 0/1.\n"
            f"Unique raw labels (sample): {list(uniq)}\n"
            f"Example rows:\n{bad[['filepath','label_raw']].to_string(index=False)}"
        )
    df["filepath"] = df["filepath"].astype(str).str.replace("\\\\","/", regex=False)
    df["basename"] = df["filepath"].apply(lambda p: Path(p).name)
    return df[["filepath","basename","label"]]

def load_preds(path):
    df = pd.read_csv(path)
    need = {"filepath","prob_anemic","pred_label","arch","tta_used"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)} (have: {list(df.columns)})")
    # Drop stray 'label' from preds if present
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df["filepath"] = df["filepath"].astype(str).str.replace("\\\\", "/", regex=False)
    df["basename"] = df["filepath"].apply(lambda p: Path(p).name)
    df["pred_label"] = df["pred_label"].astype(int)
    return df

def smart_merge(preds, labels):
    """Merge preds with labels using the best key (filepath or basename)."""
    m_full = preds.merge(labels[["filepath","label"]], on="filepath", how="left")
    miss_full = int(m_full["label"].isna().sum())
    lbl_dedup = labels.drop_duplicates(subset=["basename"])
    m_base = preds.merge(lbl_dedup[["basename","label"]], on="basename", how="left")
    miss_base = int(m_base["label"].isna().sum())
    return (m_base, "basename", miss_base) if miss_base < miss_full else (m_full, "filepath", miss_full)

def metrics(df):
    y = df["label"].astype(int); p = df["pred_label"].astype(int)
    tp = int(((y==1)&(p==1)).sum()); tn = int(((y==0)&(p==0)).sum())
    fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    n = len(df); acc = (tp+tn)/n if n else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return dict(n=n,tp=tp,tn=tn,fp=fp,fn=fn,acc=acc,prec=prec,rec=rec,f1=f1)

def dump_fp_fn(df, fp_path, fn_path):
    fp = df[(df["label"]==0) & (df["pred_label"]==1)].copy()
    fn = df[(df["label"]==1) & (df["pred_label"]==0)].copy()
    fp.to_csv(fp_path, index=False); fn.to_csv(fn_path, index=False)
    return len(fp), len(fn)

def fmt(m):
    return (f"n={m['n']}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} | "
            f"Acc={m['acc']:.4f} Prec={m['prec']:.4f} Rec={m['rec']:.4f} F1={m['f1']:.4f}")

# -------- Run --------
labels     = load_labels(LABELS)
resnet_raw = load_preds(RESNET_PRED)
mbnet_raw  = load_preds(MBNET_PRED)

resnet, key_r, miss_r = smart_merge(resnet_raw, labels)
mbnet,  key_m, miss_m = smart_merge(mbnet_raw,  labels)

if miss_r or miss_m:
    ex_r = resnet[resnet["label"].isna()]["filepath"].head(5).tolist()
    ex_m = mbnet[mbnet["label"].isna()]["filepath"].head(5).tolist()
    print(f"[WARN] Missing labels after merge → ResNet18={miss_r} (e.g. {ex_r}) | MobileNetV3={miss_m} (e.g. {ex_m})")

# drop any rows that still lack labels
resnet = resnet.dropna(subset=["label"]).copy()
mbnet  = mbnet.dropna(subset=["label"]).copy()
resnet["label"] = resnet["label"].astype(int)
mbnet["label"]  = mbnet["label"].astype(int)

# save labeled versions (drop helper basename)
resnet.drop(columns=["basename"]).to_csv(RESNET_LABELED, index=False)
mbnet.drop(columns=["basename"]).to_csv(MBNET_LABELED, index=False)

# metrics + fp/fn
r_m = metrics(resnet); m_m = metrics(mbnet)
r_fp_n, r_fn_n = dump_fp_fn(resnet, RESNET_FP, RESNET_FN)
m_fp_n, m_fn_n = dump_fp_fn(mbnet,  MBNET_FP,  MBNET_FN)

# side-by-side compare
cmp = (resnet[["filepath","label","prob_anemic","pred_label"]]
       .rename(columns={"prob_anemic":"prob_resnet18","pred_label":"pred_resnet18"})
       .merge(mbnet[["filepath","prob_anemic","pred_label"]],
              on="filepath", how="inner", suffixes=("","_m")))

# Handle both cases: with or without merge suffixes on MobileNet columns
m_pred_col = "pred_label_m"  if "pred_label_m"  in cmp.columns else "pred_label"
m_prob_col = "prob_anemic_m" if "prob_anemic_m" in cmp.columns else "prob_anemic"

cmp = cmp.rename(columns={
    m_pred_col: "pred_mobilenetv3",
    m_prob_col: "prob_mobilenetv3"
})

cmp["agree"] = (cmp["pred_resnet18"]==cmp["pred_mobilenetv3"]).astype(int)
cmp["resnet18_correct"]    = (cmp["pred_resnet18"]==cmp["label"]).astype(int)
cmp["mobilenetv3_correct"] = (cmp["pred_mobilenetv3"]==cmp["label"]).astype(int)
cmp["which_correct"] = cmp.apply(
    lambda r: "both_correct" if r["resnet18_correct"] and r["mobilenetv3_correct"]
    else ("resnet18_only" if r["resnet18_correct"] and not r["mobilenetv3_correct"]
    else ("mobilenetv3_only" if r["mobilenetv3_correct"] and not r["resnet18_correct"]
    else "both_wrong")), axis=1)

cmp.to_csv(COMPARE_CSV, index=False)

# -------- Report --------
print("== Merge keys ==")
print(f"ResNet18 merge key: {key_r} | MobileNetV3 merge key: {key_m}")

print("\n== Summary ==")
print(f"ResNet18:    {fmt(r_m)}   (FP: {r_fp_n} -> {RESNET_FP}, FN: {r_fn_n} -> {RESNET_FN})")
print(f"MobileNetV3: {fmt(m_m)}   (FP: {m_fp_n} -> {MBNET_FP},  FN: {m_fn_n} -> {MBNET_FN})")

print("\nModel agreement breakdown:")
print(cmp['which_correct'].value_counts().rename_axis('case').to_frame('count'))

print(f"\nWrote labeled CSVs:\n  - {RESNET_LABELED}\n  - {MBNET_LABELED}")
print(f"Side-by-side comparison:\n  - {COMPARE_CSV}")
