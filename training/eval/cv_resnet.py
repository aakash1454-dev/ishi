#!/usr/bin/env python3
"""
The honest ResNet18 re-baseline -- the REAL starting line.

Re-runs the current model (ImageNet ResNet18, legacy recipe) under the trusted
subject-level 5-fold split. Each fold's held-out subjects are scored by a model
that never saw them; predictions are pooled into one out-of-fold (OOF) set of all
216 subjects. Epochs are FIXED (no peeking at the held fold to pick a checkpoint),
which is what keeps the number honest -- expect it BELOW the old leaky 88% F1.

Run (GPU recommended):
  uv run python -m training.eval.cv_resnet --epochs 20
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from training.eval.dataset import CropDataset, load_folds, legacy_transforms, class_weights
from training.eval.runlog import screening_metrics, write_run


def build_resnet18(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def train_fold(train_rows, val_rows, args, device, fold):
    train_tf, eval_tf = legacy_transforms(args.img_size)
    tr = CropDataset(train_rows, train_tf)
    va = CropDataset(val_rows, eval_tf)
    # seed the sampler per (seed, fold) so distinct --seed values give cleanly
    # distinct training orders (needed for the multi-seed mean+/-std protocol).
    gen = torch.Generator().manual_seed(args.seed * 1000 + fold)
    sampler = WeightedRandomSampler(class_weights(train_rows), len(train_rows),
                                    replacement=True, generator=gen)
    tl = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=0)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=0)

    model = build_resnet18(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()

    # OOF predictions on the held-out fold (P(anemic) = softmax index 1)
    model.eval()
    p, ytrue, subs = [], [], []
    with torch.no_grad():
        for x, y, s in vl:
            prob = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            p.extend(prob.tolist()); ytrue.extend(y.tolist()); subs.extend(s)
    return subs, ytrue, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eval/resnet18_legacy")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = load_folds()
    n_folds = max(r["fold"] for r in rows) + 1

    oof = {}   # subject -> (y_true, p_anemic, fold)
    per_fold = []
    for f in range(n_folds):
        train_rows = [r for r in rows if r["fold"] != f]
        val_rows = [r for r in rows if r["fold"] == f]
        t0 = time.time()
        subs, ytrue, p = train_fold(train_rows, val_rows, args, device, f)
        for s, yt, pp in zip(subs, ytrue, p):
            oof[s] = (yt, pp, f)
        m = screening_metrics(ytrue, p)
        per_fold.append(m)
        print(f"fold {f}: F1={m['f1']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}  AUROC={m['auroc'] and round(m['auroc'],3)}  "
              f"({time.time()-t0:.0f}s)")

    subjects = sorted(oof)
    y_all = [oof[s][0] for s in subjects]
    p_all = [oof[s][1] for s in subjects]
    overall = screening_metrics(y_all, p_all)

    def ms(key):
        v = [m[key] for m in per_fold if m[key] is not None]
        return float(np.mean(v)), float(np.std(v))

    metrics = {
        "model": "resnet18_legacy",
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": "resnet18_legacy", "recipe": "legacy(ColorJitter)",
              "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "wd": args.wd,
              "img_size": args.img_size, "seed": args.seed, "n_folds": n_folds,
              "pretrained": not args.no_pretrained, "selection": "fixed-epochs (no held-fold peeking)"}
    oof_rows = [[s, oof[s][0], round(oof[s][1], 6), oof[s][2]] for s in subjects]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print("\n=== ResNet18 legacy re-baseline (OOF over all 216 subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
