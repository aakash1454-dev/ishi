#!/usr/bin/env python3
import argparse, os, pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="runs/eval/stacked_meta_labeled.csv")
    ap.add_argument("--out-dir",  default="runs/eval/error_sets")
    ap.add_argument("--uncert-band", type=float, default=0.1,
                    help="|prob-0.5| <= band -> uncertain set")
    return ap.parse_args()

def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.pred_csv)
    need = {"filepath","label","prob_anemic","pred_label"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns: {sorted(miss)}")

    fp = df[(df["label"]==0)&(df["pred_label"]==1)]
    fn = df[(df["label"]==1)&(df["pred_label"]==0)]
    un = df[df["prob_anemic"].sub(0.5).abs() <= a.uncert_band]

    fp.to_csv(os.path.join(a.out_dir,"fp.csv"), index=False)
    fn.to_csv(os.path.join(a.out_dir,"fn.csv"), index=False)
    un.to_csv(os.path.join(a.out_dir,"uncertain.csv"), index=False)

    print("Saved:")
    print(" ", os.path.join(a.out_dir,"fp.csv"))
    print(" ", os.path.join(a.out_dir,"fn.csv"))
    print(" ", os.path.join(a.out_dir,"uncertain.csv"), f"(band ±{a.uncert_band})")

if __name__ == "__main__":
    main()
