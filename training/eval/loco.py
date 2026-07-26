#!/usr/bin/env python3
"""
Leave-One-Country-Out (LOCO) -- the direct test of the population confound.

Every result so far has trained and tested on a MIX of India + Italy, which lets the
model shortcut ("looks Indian -> guess anemic", since India is 72% anemic vs Italy's
19%). LOCO removes that option entirely:

    train on India  ->  test on Italy      (never saw an Italian eye)
    train on Italy  ->  test on India      (never saw an Indian eye)

There is no shared population to exploit, so whatever score survives is real,
transferable anemia signal. This is the honest ceiling for deployment: a real user is
always "a new population" to the model.

How to read it (compare against the pooled CV numbers):
  - LOCO ~= pooled within-country  -> the model learned genuine pallor.
  - LOCO collapses toward 0.5      -> the model was leaning on population cues, and
                                      will not transfer to a new clinic/device.

Run:
  uv run python -m training.eval.loco --recipe legacy --epochs 20 --seed 42
"""
import argparse
import os

import numpy as np
import torch

from training.eval.dataset import CROPSETS, PREPROCS, RECIPES, load_folds
from training.eval.cv_resnet import train_fold, FREEZE_PREFIXES
from training.eval.runlog import screening_metrics, write_run

COUNTRIES = ["India", "Italy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES), default="legacy")
    ap.add_argument("--crops", choices=list(CROPSETS), default="rectangle")
    ap.add_argument("--preproc", choices=list(PREPROCS), default="none")
    ap.add_argument("--folds", default="data/folds_balanced.csv",
                    help="only used to read subjects/labels/country; folds ignored")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--cosine", dest="cosine", action="store_true", default=False)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune", choices=list(FREEZE_PREFIXES), default="partial")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-4)
    ap.add_argument("--lr-layer3", type=float, default=1e-5)
    # forwarded to cv_resnet.train_fold (added there after loco.py was written;
    # --crops/--preproc already defined above)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--truncate", choices=["none", "layer4"], default="none")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"runs/eval/loco_resnet18_{args.recipe}_{args.finetune}/seed{args.seed}"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_folds(args.folds)
    print(f"device: {device} | recipe: {args.recipe} | finetune: {args.finetune}")

    results = {}
    for i, held in enumerate(COUNTRIES):
        train_rows = [r for r in rows if r["country"] != held]
        test_rows = [r for r in rows if r["country"] == held]
        tr_a = sum(r["y"] for r in train_rows)
        te_a = sum(r["y"] for r in test_rows)
        other = [c for c in COUNTRIES if c != held][0]
        print(f"\n=== TRAIN on {other} (n={len(train_rows)}, {tr_a} anemic) "
              f"-> TEST on {held} (n={len(test_rows)}, {te_a} anemic) ===", flush=True)

        # reuse the exact same training path as the CV runner (fold id = i seeds sampler)
        subs, ytrue, p = train_fold(train_rows, test_rows, args, device, i)
        m = screening_metrics(ytrue, p)
        results[f"train_{other}_test_{held}"] = m
        print(f"  AUROC {m['auroc']:.3f} | sens {m['sensitivity']:.3f} | "
              f"spec {m['specificity']:.3f} | F1 {m['f1']:.3f} | "
              f"confusion {m['confusion']}")

    metrics = {"model": f"loco_resnet18_{args.recipe}", "directions": results}
    config = {"model": "loco_resnet18", "recipe": args.recipe, "crops": args.crops,
              "preproc": args.preproc, "finetune": args.finetune,
              "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "wd": args.wd,
              "img_size": args.img_size, "seed": args.seed,
              "protocol": "leave-one-country-out (no shared population)"}
    out = write_run(args.out, config, metrics)

    print("\n=== LEAVE-ONE-COUNTRY-OUT SUMMARY ===")
    for k, m in results.items():
        print(f"  {k:26s} AUROC {m['auroc']:.3f}  sens {m['sensitivity']:.3f}  "
              f"spec {m['specificity']:.3f}")
    aur = [m["auroc"] for m in results.values() if m["auroc"] is not None]
    print(f"  mean transfer AUROC: {np.mean(aur):.3f}")
    print("  (compare vs pooled within-country: India ~0.59, Italy ~0.89)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
