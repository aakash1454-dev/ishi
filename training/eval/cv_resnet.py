#!/usr/bin/env python3
"""
The honest ResNet18 CV runner -- the REAL starting line, now recipe-parameterized.

Re-runs ImageNet ResNet18 under the trusted subject-level 5-fold split. Each fold's
held-out subjects are scored by a model that never saw them; predictions are pooled
into one out-of-fold (OOF) set of all 217 subjects. Epochs are FIXED (no peeking at
the held fold to pick a checkpoint), which is what keeps the number honest.

`--recipe legacy` reproduces the current production recipe (the leaky-88% successor,
~F1 0.75). `--recipe colorhygiene` is the Phase-2 fix (gray-world WB, no sat/hue
jitter, geometric aug, label smoothing, cosine LR). Same folds either way, so the
comparison is paired. Use seeds 42/43/44 and judge by mean +/- std.

Run (GPU recommended):
  uv run python -m training.eval.cv_resnet --recipe colorhygiene --epochs 20 --seed 42
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from training.eval.dataset import (
    CROPSETS, PREPROCS, with_preproc, load_folds, RECIPES, HYGIENE_OPT_RECIPES,
    class_weights)
from training.eval.runlog import screening_metrics, write_run


def build_resnet18(pretrained=True, dropout=0.0, truncate="none"):
    """truncate='layer4' drops ResNet18's last (largest) block and pools from layer3,
    cutting the model from ~11M to ~3M parameters. Motivation: in our own tournament
    accuracy tracked INVERSELY with capacity (ResNet18 11M > ConvNeXt 28M), and layer4
    encodes high-level semantics we likely don't need to judge tissue redness."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    feat = m.fc.in_features                      # 512
    if truncate == "layer4":
        m.layer4 = nn.Identity()
        feat = 256                               # layer3 output channels
    head = [nn.Dropout(dropout)] if dropout > 0 else []
    head.append(nn.Linear(feat, 2))
    # nn.Sequential keeps the "fc." prefix, so the discriminative-LR grouping still works
    m.fc = nn.Sequential(*head) if len(head) > 1 else head[0]
    return m


# how much of the backbone stays frozen, per --finetune mode
FREEZE_PREFIXES = {
    "full":    (),
    "partial": ("conv1", "bn1", "layer1", "layer2"),
    "layer4":  ("conv1", "bn1", "layer1", "layer2", "layer3"),
    "head":    ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
}


def build_optimizer(model, args):
    """full  = every layer at one LR (the existing behaviour).
    partial = freeze the generic early layers (conv1/bn1/layer1/layer2) and give the
    rest DISCRIMINATIVE LRs -- late layers (task-specific) learn fast, early layers
    slow. Standard small-data transfer-learning move: full freezing underuses the
    data, full fine-tuning overfits 174 images."""
    if args.finetune == "full":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    frozen = FREEZE_PREFIXES[args.finetune]
    for n, p in model.named_parameters():
        if n.startswith(frozen):
            p.requires_grad = False

    def params(prefix):
        return [p for n, p in model.named_parameters()
                if n.startswith(prefix) and p.requires_grad]

    # skip empty groups: layer4 vanishes under --truncate, layer3 under --finetune head
    groups = [g for g in (
        {"params": params("layer3"), "lr": args.lr_layer3},
        {"params": params("layer4"), "lr": args.lr_layer4},
        {"params": params("fc"),     "lr": args.lr_head},
    ) if g["params"]]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  finetune={args.finetune}: training {n_train/1e6:.2f}M / {n_all/1e6:.2f}M params "
          f"in {len(groups)} LR group(s)", flush=True)
    return torch.optim.AdamW(groups, weight_decay=args.wd)


def train_fold(train_rows, val_rows, args, device, fold):
    train_tf, eval_tf = with_preproc(RECIPES[args.recipe](args.img_size), args.preproc)
    DS = CROPSETS[args.crops]          # rectangle (skin included) | masked (skin removed)
    tr = DS(train_rows, train_tf)
    va = DS(val_rows, eval_tf)
    # seed the sampler per (seed, fold) so distinct --seed values give cleanly
    # distinct training orders (needed for the multi-seed mean+/-std protocol).
    gen = torch.Generator().manual_seed(args.seed * 1000 + fold)
    sampler = WeightedRandomSampler(class_weights(train_rows), len(train_rows),
                                    replacement=True, generator=gen)
    tl = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model = build_resnet18(pretrained=not args.no_pretrained,
                           dropout=args.dropout, truncate=args.truncate).to(device)
    opt = build_optimizer(model, args)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.cosine else None)

    print(f"  fold {fold}: training {len(train_rows)} imgs, {args.epochs} epochs "
          f"(workers={args.workers}) ...", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if sched is not None:
            sched.step()
        if args.log_every and (ep + 1) % args.log_every == 0:
            print(f"    epoch {ep+1:2d}/{args.epochs}  loss={tot/max(nb,1):.4f}", flush=True)

    # OOF predictions on the held-out fold (P(anemic) = softmax index 1)
    model.eval()
    p, ytrue, subs = [], [], []
    with torch.no_grad():
        for x, y, s in vl:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1)[:, 1]
            if args.tta:
                # average P(anemic) over the image and its mirror. Horizontal flip is
                # label-preserving here (a left/right-flipped eyelid is still the same
                # eyelid) and averages away some prediction noise.
                flip = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)[:, 1]
                prob = (prob + flip) / 2
            prob = prob.cpu().numpy()
            p.extend(prob.tolist()); ytrue.extend(y.tolist()); subs.extend(s)
    return subs, ytrue, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES), default="legacy")
    ap.add_argument("--crops", choices=list(CROPSETS), default="rectangle",
                    help="rectangle = data/crops (skin included); "
                         "masked = datasets/eda-masked-crops (skin removed)")
    ap.add_argument("--preproc", choices=list(PREPROCS), default="none",
                    help="colour-space preprocessing: lab (CIELAB) | clahe (CLAHE on L only)")
    ap.add_argument("--truncate", choices=["none", "layer4"], default="none",
                    help="layer4 = drop ResNet18's last block (~11M -> ~3M params)")
    ap.add_argument("--finetune", choices=list(FREEZE_PREFIXES), default="partial",
                    help="DEFAULT partial: freeze conv1/layer1/layer2 + discriminative "
                         "LRs. Adopted after a 3-seed test beat full fine-tuning "
                         "+0.042 AUROC with non-overlapping seed ranges. "
                         "Use --finetune full for the old behaviour.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout before the classifier head (Tier-1 regularisation)")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average P(anemic) over image + hflip")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="partial mode: fc LR")
    ap.add_argument("--lr-layer4", type=float, default=1e-4, help="partial mode: layer4 LR")
    ap.add_argument("--lr-layer3", type=float, default=1e-5, help="partial mode: layer3 LR")
    ap.add_argument("--folds", default="data/folds.csv",
                    help="split file; e.g. data/folds_balanced_3.csv")
    ap.add_argument("--out", default=None,
                    help="default: runs/eval/resnet18_<recipe>[_<split>]/seed<seed>")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; keep 0 on Windows (spawn hangs)")
    ap.add_argument("--log-every", type=int, default=1,
                    help="print train loss every N epochs (0 = silent)")
    ap.add_argument("--label-smoothing", type=float, default=None,
                    help="default: 0.0 for legacy, 0.05 for colorhygiene")
    ap.add_argument("--cosine", dest="cosine", action="store_true", default=None,
                    help="cosine LR decay; on by default for colorhygiene")
    ap.add_argument("--no-cosine", dest="cosine", action="store_false")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    # Recipe-driven defaults (the colorhygiene recipe is a bundle: WB + aug + label
    # smoothing + cosine LR). Explicit flags still override.
    hygiene_opt = args.recipe in HYGIENE_OPT_RECIPES
    if args.label_smoothing is None:
        args.label_smoothing = 0.05 if hygiene_opt else 0.0
    if args.cosine is None:
        args.cosine = hygiene_opt
    # tag the output dir with the split name (unless it's the default folds.csv)
    split_tag = os.path.splitext(os.path.basename(args.folds))[0]
    suffix = "" if split_tag == "folds" else f"_{split_tag}"
    croptag = "" if args.crops == "rectangle" else f"_{args.crops}"
    imgtag = "" if args.img_size == 224 else f"_img{args.img_size}"
    pretag = "" if args.preproc == "none" else f"_{args.preproc}"
    fttag = "" if args.finetune == "full" else f"_{args.finetune}"
    trunctag = "" if args.truncate == "none" else f"_trunc{args.truncate}"
    regtag = ("" if args.dropout == 0 else f"_do{args.dropout:g}") + \
             ("" if not args.label_smoothing else f"_ls{args.label_smoothing:g}") + \
             ("_tta" if args.tta else "")
    if args.out is None:
        args.out = (f"runs/eval/resnet18_{args.recipe}{croptag}{imgtag}{pretag}{fttag}"
                    f"{trunctag}{regtag}{suffix}/seed{args.seed}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| folds:", args.folds)

    rows = load_folds(args.folds)
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

    model_name = f"resnet18_{args.recipe}"
    metrics = {
        "model": model_name,
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": model_name, "recipe": args.recipe, "crops": args.crops,
              "preproc": args.preproc, "finetune": args.finetune,
              "dropout": args.dropout, "tta": args.tta, "truncate": args.truncate,
              "lr_head": args.lr_head, "lr_layer4": args.lr_layer4,
              "lr_layer3": args.lr_layer3, "folds": args.folds,
              "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "wd": args.wd,
              "label_smoothing": args.label_smoothing, "cosine_lr": args.cosine,
              "img_size": args.img_size, "seed": args.seed, "n_folds": n_folds,
              "pretrained": not args.no_pretrained, "selection": "fixed-epochs (no held-fold peeking)"}
    oof_rows = [[s, oof[s][0], round(oof[s][1], 6), oof[s][2]] for s in subjects]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print(f"\n=== ResNet18 [{args.recipe}] (OOF over all {len(subjects)} subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
