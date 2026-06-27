#!/usr/bin/env python3
"""
The simple-baseline FLOOR. Anemia's cue is conjunctiva colour (pallor), so a
hand-built colour-feature model + logistic regression sets the bar every deep
model must clearly beat to justify its complexity. If a CNN can't beat this,
that's a finding, not a failure of the harness.

Evaluated under the SAME subject-level folds (data/folds.csv) as every other
model, out-of-fold (each subject scored by a model that never saw it). Standardiser
+ logreg are fit on train folds only (no leakage into the scaler).

Run (after make_folds):
  uv run python -m training.eval.baseline_color
"""
import argparse
import csv
import pathlib

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.color import rgb2lab, rgb2hsv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from training.eval.runlog import screening_metrics, write_run

CROPS = pathlib.Path("data/crops")
FOLDS = pathlib.Path("data/folds.csv")
SIZE = 160
EPS = 1e-6


def features(path):
    """Compact, interpretable colour descriptor for one conjunctiva crop."""
    img = np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE)),
                     dtype=np.float32) / 255.0
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    lab = rgb2lab(img); hsv = rgb2hsv(img)
    a_star = lab[..., 1]            # +a = red, -a = green  (the pallor axis)
    redness = R / (R + G + B + EPS)

    # reddest ~30% of pixels -> rough conjunctiva region (crops aren't masked)
    sel = redness >= np.quantile(redness, 0.70)
    f = [
        R.mean(), G.mean(), B.mean(), R.std(), G.std(), B.std(),
        a_star.mean(), lab[..., 2].mean(), lab[..., 0].mean(),   # a*, b*, L
        hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean(),  # H, S, V
        redness.mean(), redness.std(),
        # reddest-region (conjunctiva proxy)
        R[sel].mean(), G[sel].mean(), B[sel].mean(),
        a_star[sel].mean(), redness[sel].mean(),
        (G[sel].mean() / (R[sel].mean() + EPS)),   # pallor proxy: G/R over red region
    ]
    return np.asarray(f, dtype=np.float32)


def load_folds(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["y"] = int(r["y"]); r["fold"] = int(r["fold"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eval/baseline_color")
    ap.add_argument("--folds", default=str(FOLDS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1
    print(f"extracting colour features for {len(rows)} crops ...")
    X = np.stack([features(CROPS / r["image"]) for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])

    oof_p = np.zeros(len(rows))
    per_fold = []
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                               random_state=args.seed),
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]   # P(anemic)
        oof_p[va] = p
        m = screening_metrics(y[va], p)
        per_fold.append(m)
        print(f"  fold {f}: F1={m['f1']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}  AUROC={m['auroc']}")

    overall = screening_metrics(y, oof_p)

    def ms(key):
        v = [m[key] for m in per_fold if m[key] is not None]
        return float(np.mean(v)), float(np.std(v))

    metrics = {
        "model": "color_logreg_baseline",
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": "color_logreg_baseline", "seed": args.seed,
              "n_folds": n_folds, "n_features": int(X.shape[1]), "img_size": SIZE}
    oof_rows = [[r["subject"], r["y"], round(float(oof_p[i]), 6), r["fold"]]
                for i, r in enumerate(rows)]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print("\n=== colour baseline (out-of-fold over all 216 subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
