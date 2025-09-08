#!/usr/bin/env python3
"""
Train a stacked logistic regressor on top of:
    - prob_resnet18
    - prob_mobilenetv3
using out-of-fold predictions (no leakage), then:
    - produce calibrated stacked probabilities
    - sweep thresholds, report best-F1 and best-Acc
    - optionally pick threshold to satisfy a precision target
    - write artifacts (stacked_labeled.csv, curves, fp/fn)

Requires: pandas, numpy, scikit-learn
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

DEF_RESNET = "runs/eval/tta_resnet18_labeled.csv"
DEF_MBNET  = "runs/eval/tta_mobilenetv3_labeled.csv"
OUT_DIR    = "runs/eval"

def load_probs(path, prob_name):
    df = pd.read_csv(path)
    need = {"filepath","label","prob_anemic"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)} (have: {list(df.columns)})")
    df = df[["filepath","label","prob_anemic"]].copy()
    df["filepath"] = df["filepath"].astype(str).str.replace("\\\\","/", regex=False)
    df["basename"] = df["filepath"].apply(lambda p: Path(p).name)
    df = df.rename(columns={"prob_anemic": prob_name})
    df["label"] = df["label"].astype(int)
    return df

def merge_inputs(resnet_path, mbnet_path):
    r = load_probs(resnet_path, "prob_resnet18")
    m = load_probs(mbnet_path,  "prob_mobilenetv3")
    lbl = r[["basename","label"]].drop_duplicates(subset=["basename"])
    df  = (lbl.merge(r[["basename","prob_resnet18"]], on="basename", how="left")
              .merge(m[["basename","prob_mobilenetv3"]], on="basename", how="left"))
    # Drop any rows missing either score
    before = len(df)
    df = df.dropna(subset=["prob_resnet18","prob_mobilenetv3"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] dropped {dropped} rows with missing model probabilities")
    return df

def metrics(y_true, y_prob, thr):
    y = y_true.astype(int).values
    p = (y_prob.values >= thr).astype(int)
    tp = int(((y==1)&(p==1)).sum()); tn = int(((y==0)&(p==0)).sum())
    fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    n = len(y)
    acc = (tp+tn)/n if n else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return dict(threshold=float(thr), n=int(n), tp=tp, tn=tn, fp=fp, fn=fn,
                acc=acc, prec=prec, rec=rec, f1=f1)

def sweep_curve(y_true, y_prob):
    rows = [metrics(y_true, y_prob, t) for t in np.linspace(0,1,101)]
    return pd.DataFrame(rows)

def pick_threshold_for_precision(curve_df, min_precision):
    # among thresholds with precision >= target, pick the one with highest recall, then highest F1 as tie-breaker
    ok = curve_df[curve_df["prec"] >= min_precision].copy()
    if ok.empty:
        return None
    ok = ok.sort_values(["rec","f1","acc"], ascending=False)
    return ok.iloc[0].to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default=DEF_RESNET)
    ap.add_argument("--mbnet",  default=DEF_MBNET)
    ap.add_argument("--folds", type=int, default=5, help="Stratified K folds")
    ap.add_argument("--precision-target", type=float, default=None,
                    help="If set (e.g. 0.80), also report threshold achieving at least this precision")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = merge_inputs(args.resnet, args.mbnet)
    X = df[["prob_resnet18","prob_mobilenetv3"]].values
    y = df["label"].values

    # Out-of-fold stacked probabilities
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(
            solver="liblinear",  # stable for small feature dims
            class_weight=None,   # change to "balanced" if classes skew hard and you want recall
            max_iter=200
        )
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1]
        print(f"[fold {fold}] coef={clf.coef_.ravel().tolist()} bias={float(clf.intercept_[0]):.4f}")

    df["prob_stacked"] = oof

    # Sweep stacked probabilities
    curve = sweep_curve(df["label"], df["prob_stacked"])
    best_f1  = curve.iloc[curve["f1"].idxmax()].to_dict()
    best_acc = curve.iloc[curve["acc"].idxmax()].to_dict()
    print("\n== Stacked (logistic, OOF) ==")
    print(f"Best F1  at t={best_f1['threshold']:.2f}:  "
          f"n={int(best_f1['n'])} TP={int(best_f1['tp'])} TN={int(best_f1['tn'])} FP={int(best_f1['fp'])} FN={int(best_f1['fn'])} | "
          f"Acc={best_f1['acc']:.4f} Prec={best_f1['prec']:.4f} Rec={best_f1['rec']:.4f} F1={best_f1['f1']:.4f}")
    print(f"Best Acc at t={best_acc['threshold']:.2f}:  "
          f"n={int(best_acc['n'])} TP={int(best_acc['tp'])} TN={int(best_acc['tn'])} FP={int(best_acc['fp'])} FN={int(best_acc['fn'])} | "
          f"Acc={best_acc['acc']:.4f} Prec={best_acc['prec']:.4f} Rec={best_acc['rec']:.4f} F1={best_acc['f1']:.4f}")

    # Optional: target precision rule
    pick = None
    if args.precision_target is not None:
        pick = pick_threshold_for_precision(curve, args.precision_target)
        if pick is None:
            print(f"\n[WARN] No threshold reaches precision ≥ {args.precision_target:.2f}")
        else:
            print(f"\nPrecision target ≥ {args.precision_target:.2f}: "
                  f"t={pick['threshold']:.2f} | Acc={pick['acc']:.4f} Prec={pick['prec']:.4f} Rec={pick['rec']:.4f} F1={pick['f1']:.4f}")

    # Write artifacts
    labeled = df.copy()
    thr_use = pick["threshold"] if pick is not None else best_f1["threshold"]
    labeled["pred_stacked"] = (labeled["prob_stacked"] >= thr_use).astype(int)

    labeled_path = os.path.join(args.out_dir, "stacked_labeled.csv")
    curve_path   = os.path.join(args.out_dir, "stacked_thr_curve.csv")
    labeled.to_csv(labeled_path, index=False)
    curve.to_csv(curve_path, index=False)

    # FP/FN exports at chosen threshold
    yhat = labeled["pred_stacked"].values
    fp = labeled[(labeled["label"]==0) & (yhat==1)]
    fn = labeled[(labeled["label"]==1) & (yhat==0)]
    fp_path = os.path.join(args.out_dir, "stacked_fp.csv")
    fn_path = os.path.join(args.out_dir, "stacked_fn.csv")
    fp.to_csv(fp_path, index=False)
    fn.to_csv(fn_path, index=False)

    print("\nSaved:")
    print(f"  - Stacked labeled predictions: {labeled_path}")
    print(f"  - Stacked threshold curve:     {curve_path}")
    print(f"  - FP at chosen threshold:      {fp_path}")
    print(f"  - FN at chosen threshold:      {fn_path}")

if __name__ == "__main__":
    main()
