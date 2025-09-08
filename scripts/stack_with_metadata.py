#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

# Inputs
DEFAULT_RESNET = "runs/eval/tta_resnet18_labeled.csv"
DEFAULT_MBNET  = "runs/eval/tta_mobilenetv3_labeled.csv"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default=DEFAULT_RESNET)
    ap.add_argument("--mbnet",  default=DEFAULT_MBNET)
    ap.add_argument("--out-dir", default="runs/eval")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--penalty", default="l2", choices=["l2","none"])
    ap.add_argument("--solver", default="lbfgs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--grid", default="0:1:0.01", help="threshold sweep start:end:step")
    return ap.parse_args()

def extract_meta(path):
    path = str(path)
    view = "forniceal_palpebral" if "forniceal_palpebral" in path else "palpebral"
    country = "Italy" if "/Italy_" in path or "Italy_" in path else "India"
    return view, country

def load_inputs(resnet_csv, mbnet_csv):
    r = pd.read_csv(resnet_csv)
    m = pd.read_csv(mbnet_csv)
    need = {"filepath","label","prob_anemic"}
    for name, df in [("ResNet", r), ("MobileNet", m)]:
        miss = need - set(df.columns)
        if miss:
            raise RuntimeError(f"{name} file missing columns: {sorted(miss)} -> {list(df.columns)}")
    r = r[["filepath","label","prob_anemic"]].rename(columns={"prob_anemic":"prob_resnet"})
    m = m[["filepath","label","prob_anemic"]].rename(columns={"prob_anemic":"prob_mbnet"})
    df = r.merge(m[["filepath","prob_mbnet"]], on="filepath", how="inner")
    df["view"], df["country"] = zip(*df["filepath"].map(extract_meta))
    return df

def build_features(df, enc=None):
    X_num = df[["prob_resnet","prob_mbnet"]].to_numpy(dtype=float)
    X_cat = df[["view","country"]].astype(str)
    if enc is None:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(X_cat)
    else:
        X_cat = enc.transform(X_cat)
    X = np.hstack([X_num, X_cat])
    y = df["label"].astype(int).to_numpy()
    return X, y, enc

def sweep_metrics(y_true, p_prob, thr_grid):
    rows = []
    for t in thr_grid:
        y_pred = (p_prob >= t).astype(int)
        tn = ((y_true==0)&(y_pred==0)).sum()
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        acc = (tp+tn)/len(y_true)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        rows.append(dict(threshold=t, n=len(y_true), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                         acc=acc, prec=prec, rec=rec, f1=f1))
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_inputs(args.resnet, args.mbnet)
    X, y, enc = build_features(df)

    # OOF logistic with metadata
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(df), dtype=float)
    coefs, intercepts = [], []
    for tr, va in kf.split(X, y):
        clf = LogisticRegression(C=args.C, penalty=args.penalty if args.penalty!="none" else None,
                                 solver=args.solver, max_iter=500)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1]
        coefs.append(clf.coef_[0].tolist()); intercepts.append(float(clf.intercept_[0]))

    print("[OOF] mean coef (first 2 are probs; rest are one-hots):")
    print(np.mean(np.array(coefs), axis=0).round(4).tolist(), "bias=", round(np.mean(intercepts), 4))

    # threshold sweep
    s, e, st = [float(x) for x in args.grid.split(":")]
    thr_grid = np.arange(s, e+1e-9, st)
    curve = sweep_metrics(y, oof, thr_grid)
    best_f1  = curve.sort_values("f1", ascending=False).head(1).iloc[0]
    best_acc = curve.sort_values("acc", ascending=False).head(1).iloc[0]

    print("\n== Stacked+Meta (OOF) ==")
    print(f"Best F1  at t={best_f1.threshold:.2f}:  n={best_f1.n} TP={int(best_f1.tp)} TN={int(best_f1.tn)} FP={int(best_f1.fp)} FN={int(best_f1.fn)} | "
          f"Acc={best_f1.acc:.4f} Prec={best_f1.prec:.4f} Rec={best_f1.rec:.4f} F1={best_f1.f1:.4f}")
    print(f"Best Acc at t={best_acc.threshold:.2f}:  n={best_acc.n} TP={int(best_acc.tp)} TN={int(best_acc.tn)} FP={int(best_acc.fp)} FN={int(best_acc.fn)} | "
          f"Acc={best_acc.acc:.4f} Prec={best_acc.prec:.4f} Rec={best_acc.rec:.4f} F1={best_acc.f1:.4f}")

    # export labeled + curve
    out_labeled = os.path.join(args.out_dir, "stacked_meta_labeled.csv")
    out_curve   = os.path.join(args.out_dir, "stacked_meta_thr_curve.csv")
    out_fp      = os.path.join(args.out_dir, "stacked_meta_fp.csv")
    out_fn      = os.path.join(args.out_dir, "stacked_meta_fn.csv")

    labeled = df.copy()
    labeled["prob_stacked_meta"] = oof
    labeled["pred_stacked_meta"] = (oof >= float(best_f1.threshold)).astype(int)
    # expose standard columns for downstream tooling
    labeled["prob_anemic"] = labeled["prob_stacked_meta"]
    labeled["pred_label"]  = labeled["pred_stacked_meta"]
    labeled["arch"] = "stacked_meta"
    labeled.to_csv(out_labeled, index=False)
    curve.to_csv(out_curve, index=False)

    fp = labeled[(labeled["label"]==0) & (labeled["pred_label"]==1)]
    fn = labeled[(labeled["label"]==1) & (labeled["pred_label"]==0)]
    fp.to_csv(out_fp, index=False); fn.to_csv(out_fn, index=False)

    print("\nSaved:")
    print(f"  - Labeled: {out_labeled}")
    print(f"  - Curve:   {out_curve}")
    print(f"  - FP:      {out_fp}")
    print(f"  - FN:      {out_fn}")

if __name__ == "__main__":
    main()
