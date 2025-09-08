#!/usr/bin/env python3
import argparse, pandas as pd, os

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="runs/eval/stacked_meta_labeled.csv",
                    help="source predictions to export")
    ap.add_argument("--out-csv",  default="runs/eval/final_predictions.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override threshold; default uses 'pred_label' if present, else 0.5")
    return ap.parse_args()

def main():
    a = parse_args()
    df = pd.read_csv(a.pred_csv)
    if "filepath" not in df.columns:
        raise SystemExit("Need 'filepath' in input preds.")
    # use prob column the slicer expects
    prob_col = "prob_anemic" if "prob_anemic" in df.columns else (
               "prob_stacked_meta" if "prob_stacked_meta" in df.columns else
               "prob_stacked")
    if a.threshold is None and "pred_label" in df.columns:
        pred = df["pred_label"].astype(int).values
    else:
        thr = 0.5 if a.threshold is None else a.threshold
        pred = (df[prob_col].values >= thr).astype(int)
    out = pd.DataFrame({
        "filepath": df["filepath"],
        "prob_anemic": df[prob_col],
        "pred_label": pred
    })
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print("Wrote:", a.out_csv, f"(n={len(out)})")
    print("Positive rate:", out["pred_label"].mean().round(4))

if __name__ == "__main__":
    main()
