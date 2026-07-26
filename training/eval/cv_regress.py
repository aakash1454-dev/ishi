#!/usr/bin/env python3
"""
Hb REGRESSION -- predict the haemoglobin value instead of the anemic/non-anemic flag.

Why: the Stage-1 diagnosis showed (a) the classifier's score already tracks real Hb
(Spearman -0.64, p=7.5e-26) and (b) its errors pile up at the WHO cutoff -- 48% error
within 0.5 g/dL of the threshold vs 12% far from it. The binary label puts a hard cut
through a continuous quantity, creating an unanswerable question for the ~25% of
subjects sitting near the boundary.

Regression avoids that: learn Hb directly, then derive the yes/no decision afterwards
(and re-tune it freely, per population, without retraining).

Evaluation reports BOTH:
  - regression quality: MAE / RMSE / Pearson / Spearman on Hb
  - binary metrics derived from predicted Hb, so it is directly comparable to the
    classifier. Anemia score = -predicted_Hb (lower Hb = more anemic), and the
    hard call uses each subject's SEX-SPECIFIC WHO cutoff (M<13, F<12).

Run:
  uv run python -m training.eval.cv_regress --folds data/folds_balanced.csv --epochs 20
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from scipy.stats import pearsonr, spearmanr

from training.eval.dataset import CROPSETS, PREPROCS, RECIPES, load_folds, with_preproc
from training.eval.runlog import screening_metrics, write_run

WHO = {"M": 13.0, "F": 12.0}          # WHO adult anemia cutoffs (g/dL)


def load_hb(path="data/manifest.csv"):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["hb"].strip():
            out[f"{r['country']}_{r['subject_id']}"] = float(r["hb"])
    return out


def build_regressor(pretrained=True, dropout=0.0):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    head = [nn.Dropout(dropout)] if dropout > 0 else []
    head.append(nn.Linear(m.fc.in_features, 1))       # single continuous output
    m.fc = nn.Sequential(*head)
    return m


def build_optimizer(model, args):
    """Same partial-fine-tune scheme adopted for the classifier (+0.042 AUROC)."""
    if args.finetune == "full":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for n, p in model.named_parameters():
        if n.startswith(("conv1", "bn1", "layer1", "layer2")):
            p.requires_grad = False
    sel = lambda pre: [p for n, p in model.named_parameters()
                       if n.startswith(pre) and p.requires_grad]
    return torch.optim.AdamW(
        [{"params": sel("layer3"), "lr": args.lr_layer3},
         {"params": sel("layer4"), "lr": args.lr_layer4},
         {"params": sel("fc"),     "lr": args.lr_head}], weight_decay=args.wd)


def train_fold(train_rows, val_rows, args, device, fold):
    train_tf, eval_tf = with_preproc(RECIPES[args.recipe](args.img_size), args.preproc)
    DS = CROPSETS[args.crops]
    tr, va = DS(train_rows, train_tf), DS(val_rows, eval_tf)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    # standardise the target using TRAIN-FOLD stats only (no leakage into val)
    ytr = np.array([r["y"] for r in train_rows], dtype=np.float32)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-6)

    model = build_regressor(pretrained=not args.no_pretrained,
                            dropout=args.dropout).to(device)
    opt = build_optimizer(model, args)
    crit = nn.SmoothL1Loss()                      # Huber: robust to Hb outliers
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.cosine else None)

    print(f"  fold {fold}: {len(train_rows)} train, {args.epochs} epochs "
          f"(Hb mean {mu:.2f} +/- {sd:.2f})", flush=True)
    for ep in range(args.epochs):
        model.train(); tot = nb = 0
        for x, yb, _ in tl:
            x = x.to(device)
            t = ((yb.float() - mu) / sd).to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), t)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if sched: sched.step()
        if args.log_every and (ep + 1) % args.log_every == 0:
            print(f"    epoch {ep+1:2d}/{args.epochs}  loss={tot/max(nb,1):.4f}", flush=True)

    model.eval(); pred, true, subs = [], [], []
    with torch.no_grad():
        for x, yb, s in vl:
            out = model(x.to(device)).squeeze(1).cpu().numpy() * sd + mu   # back to g/dL
            pred.extend(out.tolist()); true.extend(yb.float().tolist()); subs.extend(s)
    return subs, true, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES), default="legacy")
    ap.add_argument("--crops", choices=list(CROPSETS), default="rectangle")
    ap.add_argument("--preproc", choices=list(PREPROCS), default="none")
    ap.add_argument("--folds", default="data/folds_balanced.csv")
    ap.add_argument("--country", choices=["all", "India", "Italy"], default="all",
                    help="restrict BOTH train and test to one population. Tests whether "
                         "a per-population model beats one mixed model (LOCO showed "
                         "Italy-only training ranks Indian patients better than mixed).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--cosine", action="store_true", default=False)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune", choices=["full", "partial"], default="partial")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-4)
    ap.add_argument("--lr-layer3", type=float, default=1e-5)
    args = ap.parse_args()

    ctag = "" if args.country == "all" else f"_{args.country}"
    if args.out is None:
        args.out = f"runs/eval/regress_resnet18_{args.finetune}{ctag}/seed{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hb = load_hb()
    rows = [r for r in load_folds(args.folds) if r["subject"] in hb]
    if args.country != "all":
        rows = [r for r in rows if r["country"] == args.country]
        print(f"PER-POPULATION MODEL: {args.country} only "
              f"({len(rows)} subjects; train folds will be ~{int(len(rows)*0.8)})")
    meta = {r["subject"]: (r["country"], r["sex"], r["y"]) for r in rows}
    for r in rows:
        r["y"] = hb[r["subject"]]          # target becomes Hb (dataset reads r["y"])
    n_folds = max(r["fold"] for r in rows) + 1
    print(f"device: {device} | {len(rows)} subjects with Hb | {n_folds} folds")

    oof = {}
    for f in range(n_folds):
        t0 = time.time()
        subs, true, pred = train_fold([r for r in rows if r["fold"] != f],
                                      [r for r in rows if r["fold"] == f], args, device, f)
        for s, t, p in zip(subs, true, pred):
            oof[s] = (t, p, f)
        mae = np.mean([abs(t - p) for t, p in zip(true, pred)])
        print(f"  fold {f}: MAE {mae:.3f} g/dL  ({time.time()-t0:.0f}s)", flush=True)

    subs = sorted(oof)
    true = np.array([oof[s][0] for s in subs]); pred = np.array([oof[s][1] for s in subs])
    ctry = np.array([meta[s][0] for s in subs]); sex = np.array([meta[s][1] for s in subs])
    ybin = np.array([meta[s][2] for s in subs])
    cut = np.array([WHO[x] for x in sex])

    def report(lab, m):
        mae = float(np.mean(np.abs(true[m] - pred[m])))
        rmse = float(np.sqrt(np.mean((true[m] - pred[m]) ** 2)))
        pr = float(pearsonr(true[m], pred[m])[0]); sr = float(spearmanr(true[m], pred[m])[0])
        # binary metrics derived from predicted Hb (score = -Hb so higher = more anemic)
        b_auc = screening_metrics(ybin[m], -pred[m])["auroc"]
        yp = (pred[m] < cut[m]).astype(int)      # per-subject sex-specific WHO cutoff
        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
        tn, fp, fn, tp = confusion_matrix(ybin[m], yp, labels=[0, 1]).ravel()
        out = dict(n=int(m.sum()), mae=mae, rmse=rmse, pearson=pr, spearman=sr,
                   auroc=b_auc, f1=float(f1_score(ybin[m], yp, zero_division=0)),
                   acc=float(accuracy_score(ybin[m], yp)),
                   sens=float(tp / (tp + fn)) if tp + fn else 0.0,
                   spec=float(tn / (tn + fp)) if tn + fp else 0.0,
                   confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
        print(f"  {lab:6s} n={out['n']:3d} | MAE {mae:.2f} RMSE {rmse:.2f} g/dL | "
              f"r={pr:+.3f} rho={sr:+.3f} | AUROC {b_auc:.3f} | "
              f"sens {out['sens']:.3f} spec {out['spec']:.3f} F1 {out['f1']:.3f}")
        return out

    print("\n=== Hb REGRESSION (out-of-fold) ===")
    print("  binary metrics derived by applying each subject's WHO cutoff to predicted Hb")
    res = {}
    for lab, m in [("ALL", np.ones(len(subs), bool)),
                   ("India", ctry == "India"), ("Italy", ctry == "Italy")]:
        if m.sum() == 0 or len(set(ybin[m])) < 2:
            continue                      # slice absent (per-population run) or single-class
        res[lab] = report(lab, m)

    metrics = {"model": "resnet18_hb_regression", "slices": res}
    config = {"model": "resnet18_hb_regression", "target": "hb (g/dL)",
              "recipe": args.recipe, "crops": args.crops, "preproc": args.preproc,
              "finetune": args.finetune, "dropout": args.dropout,
              "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "wd": args.wd,
              "img_size": args.img_size, "seed": args.seed, "folds": args.folds,
              "binary_rule": "predicted_hb < WHO cutoff (M 13.0 / F 12.0)"}
    oof_rows = [[s, round(oof[s][0], 3), round(oof[s][1], 3), oof[s][2]] for s in subs]
    out = write_run(args.out, config, metrics, oof=oof_rows)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
