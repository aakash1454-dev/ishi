#!/usr/bin/env python3
"""
DINOv2 ViT-S/14 frozen-feature linear probe -- the strong small-data ViT path.

Self-supervised DINOv2 features without fine-tuning: run each crop through the FROZEN
backbone ONCE, cache the 384-dim feature, then fit StandardScaler + logistic regression
per fold. This is the same "frozen features -> logreg -> OOF CV" shape as
baseline_color.py, so the comparison against the colour FLOOR is apples-to-apples --
the only thing that changes is hand-built colour stats vs learned DINOv2 features.

Why this design (matches the plan):
- Frozen + cached -> fast even on CPU, and a strong regularizer on 217 subjects.
- A linear probe is the canonical frozen-DINOv2 eval and the least overfit-prone head.
- Features are deterministic, so unlike the CNN this is ~seed-invariant; one run suffices
  (a --seed knob is kept only for the logreg/parity with the rest of the harness).

The locked Phase-2 finding (gray-world WB HURTS) carries over: we use DINOv2's own
ImageNet-norm preprocessing, NO white balance, NO train-time aug (single cached view).

Run (after `pip install -r requirements_dev.txt`):
  uv run python -m training.eval.cv_dinov2
"""
import argparse
import csv
import pathlib

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from training.eval.runlog import screening_metrics, write_run

CROPS = pathlib.Path("data/crops")
FOLDS = pathlib.Path("data/folds.csv")
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"
CACHE_DIR = pathlib.Path("runs/eval/_cache")


def load_folds(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["y"] = int(r["y"]); r["fold"] = int(r["fold"])
    return rows


def build_extractor(model_name, device):
    """Frozen DINOv2 backbone (num_classes=0 -> pooled feature) + its canonical
    eval transform (correct resize/crop/normalisation for the model)."""
    import timm
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf


def extract_features(rows, model_name, device, cache_path):
    """Return X (N, D) of frozen features, aligned to rows. Cached to disk keyed by
    model + image list so seeds/re-runs are instant; only missing images are computed."""
    names = [r["image"] for r in rows]
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cache = {n: v for n, v in zip(z["images"], z["feats"])}
    missing = [n for n in names if n not in cache]
    if missing:
        print(f"extracting DINOv2 features for {len(missing)} crops "
              f"(cached {len(names) - len(missing)}) ...")
        model, tf = build_extractor(model_name, device)
        with torch.no_grad():
            for i, n in enumerate(missing):
                img = Image.open(CROPS / n).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                cache[n] = model(x).squeeze(0).cpu().numpy().astype(np.float32)
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(missing)}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        alln = sorted(cache)
        np.savez(cache_path, images=np.array(alln),
                 feats=np.stack([cache[n] for n in alln]))
    else:
        print(f"all {len(names)} DINOv2 features loaded from cache")
    return np.stack([cache[n] for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eval/dinov2_s14_linprobe")
    ap.add_argument("--folds", default=str(FOLDS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--C", type=float, default=1.0, help="logreg inverse-reg strength")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| model:", args.model)

    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1
    cache_path = CACHE_DIR / f"{args.model.replace('/', '_')}.npz"
    X = extract_features(rows, args.model, device, cache_path)
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    print(f"feature matrix: {X.shape}")

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
        p = clf.predict_proba(X[va])[:, 1]   # P(anemic)
        oof_p[va] = p
        m = screening_metrics(y[va], p)
        per_fold.append(m)
        print(f"  fold {f}: F1={m['f1']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}  AUROC={m['auroc'] and round(m['auroc'], 3)}")

    overall = screening_metrics(y, oof_p)

    def ms(key):
        v = [m[key] for m in per_fold if m[key] is not None]
        return float(np.mean(v)), float(np.std(v))

    model_name = "dinov2_s14_linprobe"
    metrics = {
        "model": model_name,
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": model_name, "backbone": args.model, "head": "logreg(frozen-probe)",
              "C": args.C, "seed": args.seed, "n_folds": n_folds,
              "feat_dim": int(X.shape[1]), "white_balance": False, "train_aug": False,
              "selection": "frozen features, no held-fold peeking"}
    oof_rows = [[r["subject"], r["y"], round(float(oof_p[i]), 6), r["fold"]]
                for i, r in enumerate(rows)]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print(f"\n=== DINOv2 ViT-S/14 frozen probe (OOF over all {len(rows)} subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
