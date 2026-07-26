#!/usr/bin/env python3
"""
Colour (hand-built) (+) DINOv2 frozen features -> one logistic regression. The
decisive cheap test: do learned deep features add ANYTHING on top of raw colour?

Both feature sets are reused verbatim from their own runners (so this is a true
superset, not a re-implementation):
  - colour: the 20-dim descriptor from baseline_color.features (the FLOOR, AUROC 0.890)
  - DINOv2: the 384-dim frozen ViT-S feature from cv_dinov2.extract_features (cached)

Same subject-level folds, same StandardScaler+logreg, OOF. Read against the floor:
  - stack ~= 0.890  -> DINOv2 features are REDUNDANT here; colour is the signal.
  - stack  > 0.890  -> complementary signal exists; an ensemble is worth building.

Run (after cv_dinov2 has cached its features, else this extracts them):
  uv run python -m training.eval.stack_color_dino
"""
import argparse
import csv
import pathlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from training.eval.runlog import screening_metrics, write_run
from training.eval.baseline_color import features as color_features
from training.eval.cv_dinov2 import extract_features, DEFAULT_MODEL, CACHE_DIR

CROPS = pathlib.Path("data/crops")
FOLDS = pathlib.Path("data/folds.csv")


def load_folds(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["y"] = int(r["y"]); r["fold"] = int(r["fold"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eval/stack_color_dino")
    ap.add_argument("--folds", default=str(FOLDS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--C", type=float, default=1.0, help="logreg inverse-reg strength")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--color-only", action="store_true",
                    help="ablate: drop DINOv2 (sanity check vs the floor)")
    ap.add_argument("--dino-only", action="store_true",
                    help="ablate: drop colour (parity check vs cv_dinov2)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1

    print(f"colour features for {len(rows)} crops ...")
    Xc = np.stack([color_features(CROPS / r["image"]) for r in rows])
    cache_path = CACHE_DIR / f"{args.model.replace('/', '_')}.npz"
    Xd = extract_features(rows, args.model, device, cache_path)

    if args.color_only:
        X, tag = Xc, "color_only"
    elif args.dino_only:
        X, tag = Xd, "dino_only"
    else:
        X, tag = np.concatenate([Xc, Xd], axis=1), "color+dino"
    print(f"[{tag}] feature matrix: {X.shape}  (colour {Xc.shape[1]} + dino {Xd.shape[1]})")

    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])

    oof_p = np.zeros(len(rows))
    per_fold = []
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=args.C,
                               random_state=args.seed),
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]
        oof_p[va] = p
        m = screening_metrics(y[va], p)
        per_fold.append(m)
        print(f"  fold {f}: F1={m['f1']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}  AUROC={m['auroc'] and round(m['auroc'], 3)}")

    overall = screening_metrics(y, oof_p)

    def ms(key):
        v = [m[key] for m in per_fold if m[key] is not None]
        return float(np.mean(v)), float(np.std(v))

    model_name = f"stack_{tag}"
    metrics = {
        "model": model_name,
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": model_name, "features": tag, "backbone": args.model,
              "n_color": int(Xc.shape[1]), "n_dino": int(Xd.shape[1]),
              "n_total": int(X.shape[1]), "C": args.C, "seed": args.seed,
              "n_folds": n_folds, "head": "logreg"}
    oof_rows = [[r["subject"], r["y"], round(float(oof_p[i]), 6), r["fold"]]
                for i, r in enumerate(rows)]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print(f"\n=== stack [{tag}] (OOF over all {len(rows)} subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}   (colour floor = 0.890)")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
