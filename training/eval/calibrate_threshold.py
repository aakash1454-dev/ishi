#!/usr/bin/env python3
"""
Stage 5 -- operating-point calibration (analysis only, no training).

LOCO showed the decision threshold does NOT transfer across populations: a model
trained on one site and applied to another either missed ~87% of cases or flagged
~70% of healthy people, even though its RANKING was fine. That is a calibration
problem, not a discrimination problem -- and calibration is fixable without
retraining.

This script reads existing out-of-fold predictions and answers:
  1. What does the sensitivity/specificity trade-off actually look like?
  2. What threshold hits a target sensitivity (screening cares about missed cases)?
  3. How much does a PER-POPULATION threshold beat one global threshold?

Run:
  uv run python -m training.eval.calibrate_threshold \
      --run runs/eval/resnet18_legacy_partial_folds_balanced --seeds 42 43 44
"""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import confusion_matrix


def load_oof(run, seeds):
    """Average P(anemic) per subject across seeds (more stable than one run)."""
    preds = {}
    for s in seeds:
        f = os.path.join(run, f"seed{s}", "oof_predictions.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            preds.setdefault(r["subject"], []).append(float(r["p_anemic"]))
    return {k: float(np.mean(v)) for k, v in preds.items()}


def sens_spec(y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return (tp / (tp + fn) if tp + fn else 0.0,
            tn / (tn + fp) if tn + fp else 0.0, (tn, fp, fn, tp))


def best_threshold(y, p, target_sens):
    """Lowest-cost threshold that still reaches the target sensitivity.
    Screening rule: a missed case is worse than a false alarm, so we fix
    sensitivity and take the best specificity available at that level."""
    best = None
    for thr in np.unique(np.round(p, 4)):
        se, sp, _ = sens_spec(y, p, thr)
        if se >= target_sens and (best is None or sp > best[2]):
            best = (thr, se, sp)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/eval/resnet18_legacy_partial_folds_balanced")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--folds", default="data/folds_balanced.csv")
    ap.add_argument("--target-sens", type=float, default=0.90)
    args = ap.parse_args()

    meta = {r["subject"]: (r["country"], int(r["y"]))
            for r in csv.DictReader(open(args.folds))}
    pr = load_oof(args.run, args.seeds)
    subs = [s for s in pr if s in meta]
    p = np.array([pr[s] for s in subs])
    y = np.array([meta[s][1] for s in subs])
    c = np.array([meta[s][0] for s in subs])
    print(f"{args.run}\n  {len(subs)} subjects, seeds {args.seeds} averaged\n")

    print("=== 1. Sensitivity / specificity trade-off (all subjects) ===")
    print(f"  {'thr':>5} {'sens':>7} {'spec':>7}   (TN,FP,FN,TP)")
    for thr in [0.2, 0.3, 0.303, 0.4, 0.5, 0.6, 0.7]:
        se, sp, cm = sens_spec(y, p, thr)
        print(f"  {thr:>5.3f} {se:>7.3f} {sp:>7.3f}   {cm}")

    print(f"\n=== 2. Threshold needed for {args.target_sens:.0%} sensitivity ===")
    for lab, m in [("ALL", np.ones(len(y), bool)),
                   ("India", c == "India"), ("Italy", c == "Italy")]:
        b = best_threshold(y[m], p[m], args.target_sens)
        if b is None:
            print(f"  {lab:6s} cannot reach {args.target_sens:.0%} sensitivity at any threshold")
        else:
            thr, se, sp = b
            print(f"  {lab:6s} threshold {thr:.3f} -> sens {se:.3f}, spec {sp:.3f}")

    print("\n=== 3. Global vs per-population threshold ===")
    g = best_threshold(y, p, args.target_sens)
    gthr = g[0] if g else 0.5
    print(f"  Global threshold tuned on everyone: {gthr:.3f}")
    print(f"  {'group':6s} {'global thr':>18} {'own thr':>22}")
    for lab, m in [("India", c == "India"), ("Italy", c == "Italy")]:
        se_g, sp_g, _ = sens_spec(y[m], p[m], gthr)
        b = best_threshold(y[m], p[m], args.target_sens)
        if b is None:
            print(f"  {lab:6s} sens {se_g:.3f}/spec {sp_g:.3f}   -- unreachable --")
            continue
        thr, se, sp = b
        print(f"  {lab:6s} sens {se_g:.3f}/spec {sp_g:.3f}   "
              f"thr {thr:.3f}: sens {se:.3f}/spec {sp:.3f}  "
              f"(spec {sp - sp_g:+.3f})")
    print("\n  -> a positive specificity delta means the population needs its OWN")
    print("     threshold; one global cutoff is leaving accuracy on the table.")


if __name__ == "__main__":
    main()
