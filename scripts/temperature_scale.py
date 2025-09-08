#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv",  default="runs/eval/stacked_meta_labeled.csv")
    ap.add_argument("--out-csv", default="runs/eval/stacked_meta_labeled_calibrated.csv")
    return ap.parse_args()

def main():
    a = parse_args()
    df = pd.read_csv(a.in_csv)
    probs = df["prob_anemic"].clip(1e-6, 1-1e-6).to_numpy()
    y     = df["label"].astype(int).to_numpy()

    def nll(T):
        p = 1/(1+np.exp(-(np.log(probs/(1-probs))/T)))
        eps=1e-12
        return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))

    res = minimize(nll, x0=[1.0], bounds=[(0.05,20.0)])
    T = float(res.x[0])

    z = np.log(probs/(1-probs))/T
    p_cal = 1/(1+np.exp(-z))
    out = df.copy()
    out["prob_anemic"] = p_cal
    out["pred_label"]  = (p_cal >= 0.5).astype(int)  # or your chosen T
    out.to_csv(a.out_csv, index=False)

    print(f"Fitted temperature T={T:.3f}")
    print("Brier (pre):", brier_score_loss(y, probs))
    print("Brier (cal):", brier_score_loss(y, p_cal))
    print("Wrote:", a.out_csv)

if __name__ == "__main__":
    main()
