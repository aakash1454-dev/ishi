#!/usr/bin/env python3
"""
D2 -- learning curves: is Italy-trained better because of QUANTITY, COMPOSITION,
or the images themselves?

Observed: a model trained on Italy ranks Indian patients better (0.689) than one
trained on India (0.595) or on the mix (0.591-0.612). Three explanations survive
the free diagnostics (image quality and Hb-range were both falsified):

  A1 quantity    - Italy-trained saw 122 subjects, India-only ~75
  A3 composition - India has only 27 healthy subjects vs Italy's 99, so an
                   India-trained model barely learns what "healthy" looks like
  A5 the site    - something about the images themselves

This script always TESTS ON INDIA (the population we care about and the one the
model fails on) and varies only the training set:

  --match total     sample N subjects, keeping each source's natural class ratio
                    -> controls QUANTITY. If Italy still wins at equal N, A1 is out.
  --match balanced  sample N/2 anemic + N/2 healthy from BOTH sources, so the
                    training sets are compositionally IDENTICAL
                    -> controls COMPOSITION. If the gap closes here, A3 was the cause
                       and the fix is "recruit more healthy Indian subjects".
                       If Italy still wins, it is A5 (the site).

Test folds come from the shared balanced split, so no training subject is ever
tested on. Results are pooled out-of-fold across all India folds.

Run:
  uv run python -m training.eval.learning_curve --match total --sizes 20 40 60
  uv run python -m training.eval.learning_curve --match balanced --sizes 20 30 40
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from training.eval.dataset import CROPSETS, PREPROCS, RECIPES, load_folds
from training.eval.cv_resnet import train_fold
from training.eval.runlog import write_run

TEST_COUNTRY = "India"


def sample_pool(pool, n, mode, rng):
    """Draw a training subset of size n. Returns None if the pool can't supply it."""
    if mode == "total":
        if len(pool) < n:
            return None
        idx = rng.choice(len(pool), n, replace=False)
        return [pool[i] for i in idx]
    # balanced: n/2 anemic + n/2 healthy, so both sources look identical
    k = n // 2
    an = [r for r in pool if r["y"] == 1]
    he = [r for r in pool if r["y"] == 0]
    if len(an) < k or len(he) < k:
        return None
    ia = rng.choice(len(an), k, replace=False)
    ih = rng.choice(len(he), k, replace=False)
    return [an[i] for i in ia] + [he[i] for i in ih]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", choices=["total", "balanced"], default="total")
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--folds", default="data/folds_balanced.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--recipe", choices=list(RECIPES), default="legacy")
    ap.add_argument("--crops", choices=list(CROPSETS), default="rectangle")
    ap.add_argument("--preproc", choices=list(PREPROCS), default="none")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)      # quiet: many runs
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--cosine", action="store_true", default=False)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune", choices=["full", "partial"], default="partial")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-4)
    ap.add_argument("--lr-layer3", type=float, default=1e-5)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"runs/eval/learning_curve_{args.match}/seed{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1

    india = [r for r in rows if r["country"] == TEST_COUNTRY]
    italy = [r for r in rows if r["country"] != TEST_COUNTRY]
    print(f"device: {device} | match={args.match} | sizes={args.sizes}")
    print(f"test always on {TEST_COUNTRY} ({len(india)} subj, "
          f"{sum(r['y'] for r in india)} anemic / {sum(1-r['y'] for r in india)} healthy)")
    print(f"Italy pool: {len(italy)} subj "
          f"({sum(r['y'] for r in italy)} anemic / {sum(1-r['y'] for r in italy)} healthy)\n")

    oof = {}            # (source, n) -> {subject: (y, p)}
    for f in range(n_folds):
        test_rows = [r for r in india if r["fold"] == f]
        pools = {"India": [r for r in india if r["fold"] != f], "Italy": italy}
        for source, pool in pools.items():
            for n in args.sizes:
                rng = np.random.default_rng(args.seed * 1000 + f * 10 + n)
                tr = sample_pool(pool, n, args.match, rng)
                if tr is None:
                    continue
                t0 = time.time()
                subs, y, p = train_fold(tr, test_rows, args, device, f)
                key = (source, n)
                oof.setdefault(key, {})
                for s, yy, pp in zip(subs, y, p):
                    oof[key][s] = (yy, pp)
                print(f"  fold {f} | train {source:5s} n={n:3d} "
                      f"(anemic {sum(r['y'] for r in tr)}/healthy {sum(1-r['y'] for r in tr)}) "
                      f"-> test {len(test_rows)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== LEARNING CURVE (match={args.match}) -- AUROC on held-out {TEST_COUNTRY} ===")
    print(f"  {'train N':>8} {'India-trained':>15} {'Italy-trained':>15} {'gap':>8}")
    results = {}
    for n in args.sizes:
        line = {}
        for source in ["India", "Italy"]:
            d = oof.get((source, n))
            if not d:
                line[source] = None; continue
            y = np.array([v[0] for v in d.values()]); p = np.array([v[1] for v in d.values()])
            line[source] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
        results[n] = line
        a = line.get("India"); b = line.get("Italy")
        fa = f"{a:.3f}" if a is not None else "  -  "
        fb = f"{b:.3f}" if b is not None else "  -  "
        gap = f"{b-a:+.3f}" if (a is not None and b is not None) else "  -  "
        print(f"  {n:>8} {fa:>15} {fb:>15} {gap:>8}")

    print("\n  Interpretation:")
    if args.match == "total":
        print("   gap stays positive at equal N -> quantity (A1) is NOT the cause")
        print("   gap closes at equal N         -> it was just sample size")
    else:
        print("   gap closes with identical composition -> A3: India lacks HEALTHY examples")
        print("   gap persists                          -> A5: something about the site itself")

    config = {"model": "learning_curve", "match": args.match, "sizes": args.sizes,
              "test_country": TEST_COUNTRY, "recipe": args.recipe, "crops": args.crops,
              "finetune": args.finetune, "epochs": args.epochs, "seed": args.seed,
              "folds": args.folds}
    out = write_run(args.out, config, {"auroc_by_size": {str(k): v for k, v in results.items()}})
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
