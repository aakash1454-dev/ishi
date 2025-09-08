#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ---------- Defaults (edit paths if yours differ) ----------
DEF_RESNET = "runs/eval/tta_resnet18_labeled.csv"
DEF_MBNET  = "runs/eval/tta_mobilenetv3_labeled.csv"
OUT_DIR    = "runs/eval"

def load_and_prepare(path, prob_name):
    df = pd.read_csv(path)
    # expected cols: filepath,label,prob_anemic,pred_label,arch,tta_used
    need = {"filepath","label","prob_anemic"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)} (have: {list(df.columns)})")
    df = df[["filepath","label","prob_anemic"]].copy()
    df["filepath"] = df["filepath"].astype(str).str.replace("\\\\", "/", regex=False)
    df["basename"] = df["filepath"].apply(lambda p: Path(p).name)
    df = df.rename(columns={"prob_anemic": prob_name})
    return df

def metrics(y_true, prob, thr):
    y = y_true.astype(int).values
    p = (prob.values >= thr).astype(int)
    tp = int(((y==1)&(p==1)).sum()); tn = int(((y==0)&(p==0)).sum())
    fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    n = len(y)
    acc = (tp+tn)/n if n else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return dict(threshold=thr, n=n, tp=tp, tn=tn, fp=fp, fn=fn,
                acc=acc, prec=prec, rec=rec, f1=f1)

def fmt(m):
    return (f"n={m['n']}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} | "
            f"Acc={m['acc']:.4f} Prec={m['prec']:.4f} Rec={m['rec']:.4f} F1={m['f1']:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default=DEF_RESNET, help="Path to resnet labeled CSV")
    ap.add_argument("--mbnet",  default=DEF_MBNET,  help="Path to mobilenet labeled CSV")
    ap.add_argument("--w-resnet", type=float, default=0.5, help="Weight for ResNet prob")
    ap.add_argument("--w-mbnet",  type=float, default=0.5, help="Weight for MobileNet prob")
    ap.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    r = load_and_prepare(args.resnet, "prob_resnet18")
    m = load_and_prepare(args.mbnet,  "prob_mobilenetv3")

    # Merge (prefer basename to ignore differing folder roots)
    lbl = r[["basename","label"]].copy()
    lbl = lbl.drop_duplicates(subset=["basename"])
    df  = (lbl.merge(r[["basename","prob_resnet18"]], on="basename", how="left")
              .merge(m[["basename","prob_mobilenetv3"]], on="basename", how="left"))

    # Any missing?
    miss_r = int(df["prob_resnet18"].isna().sum())
    miss_m = int(df["prob_mobilenetv3"].isna().sum())
    if miss_r or miss_m:
        ex_r = df[df["prob_resnet18"].isna()]["basename"].head(5).tolist()
        ex_m = df[df["prob_mobilenetv3"].isna()]["basename"].head(5).tolist()
        print(f"[WARN] Missing predictions → ResNet18={miss_r} (e.g. {ex_r}) | "
              f"MobileNetV3={miss_m} (e.g. {ex_m})")
    df = df.dropna(subset=["prob_resnet18","prob_mobilenetv3"]).copy()
    df["label"] = df["label"].astype(int)

    # Weighted average ensemble
    w_r, w_m = args.w_resnet, args.w_mbnet
    if (w_r + w_m) == 0:
        raise ValueError("w-resnet + w-mbnet must be > 0")
    s = w_r + w_m
    w_r /= s; w_m /= s
    df["prob_avg"] = w_r*df["prob_resnet18"] + w_m*df["prob_mobilenetv3"]

    # Threshold sweep
    rows = []
    for t in np.linspace(0, 1, 101):
        rows.append(metrics(df["label"], df["prob_avg"], t))
    curve = pd.DataFrame(rows)

    # Pick bests
    best_f1 = curve.iloc[curve["f1"].idxmax()]
    best_acc = curve.iloc[curve["acc"].idxmax()]

    # Save artifacts
    curve_path = os.path.join(args.out_dir, "ensemble_thr_curve.csv")
    df_out = df.copy()
    df_out["pred_avg@bestF1"] = (df_out["prob_avg"] >= best_f1["threshold"]).astype(int)
    labeled_path = os.path.join(args.out_dir, "ensemble_labeled.csv")
    df_out.to_csv(labeled_path, index=False)
    curve.to_csv(curve_path, index=False)

    # Quick FP/FN at best-F1 threshold
    y = df_out["label"].values
    p = df_out["pred_avg@bestF1"].values
    fp_idx = (y==0) & (p==1)
    fn_idx = (y==1) & (p==0)
    fp_csv = os.path.join(args.out_dir, "ensemble_fp.csv")
    fn_csv = os.path.join(args.out_dir, "ensemble_fn.csv")
    df_out.loc[fp_idx, ["basename","label","prob_resnet18","prob_mobilenetv3","prob_avg","pred_avg@bestF1"]].to_csv(fp_csv, index=False)
    df_out.loc[fn_idx, ["basename","label","prob_resnet18","prob_mobilenetv3","prob_avg","pred_avg@bestF1"]].to_csv(fn_csv, index=False)

    # Report
    print("== Ensemble (weighted average) ==")
    print(f"Weights: ResNet18={w_r:.3f}, MobileNetV3={w_m:.3f}")
    print(f"Best F1 at t={best_f1['threshold']:.2f}:  {fmt(best_f1)}")
    print(f"Best Acc at t={best_acc['threshold']:.2f}: {fmt(best_acc)}")
    print(f"\nSaved:")
    print(f"  - Labeled ensemble predictions: {labeled_path}")
    print(f"  - Threshold curve:              {curve_path}")
    print(f"  - FP at best-F1:                {fp_csv}")
    print(f"  - FN at best-F1:                {fn_csv}")

if __name__ == "__main__":
    main()
