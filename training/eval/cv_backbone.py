#!/usr/bin/env python3
"""
Generalized timm-backbone fine-tune CV runner -- the supervised deep models with the
best shot at the colour floor (they RETAIN colour, unlike frozen DINOv2, and we
fine-tune end-to-end). Same trusted subject-level folds + locked colorhygiene_nowb
recipe as everything else, so the comparison stays paired.

Mirrors cv_resnet.py exactly (sampler, fixed-epoch selection, label smoothing, cosine
LR, OOF pooling) but builds any timm backbone with a 2-class head via --backbone. Keep
cv_resnet.py for the torchvision ResNet18 baseline; use this for the new backbones.

Run (GPU):
  uv run python -m training.eval.cv_backbone --backbone convnext_tiny --seed 42
  uv run python -m training.eval.cv_backbone --backbone tf_efficientnetv2_s --seed 42
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.eval.dataset import (
    CropDataset, load_folds, RECIPES, HYGIENE_OPT_RECIPES, class_weights)
from training.eval.runlog import screening_metrics, write_run


def build_backbone(name, pretrained=True):
    """timm model with a fresh 2-class head (timm handles the classifier swap)."""
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=2)


def build_optimizer(model, args):
    """Generic equivalent of the ResNet partial fine-tune (which gave +0.042).

    timm backbones don't share ResNet's layer1..layer4 naming (ConvNeXt uses
    stages.0-3, EfficientNetV2 uses blocks.0-6), so instead of matching names we
    freeze the first `--freeze-frac` of the backbone's parameters *in forward order*
    and give the classifier its own higher LR. Same idea, architecture-agnostic:
    keep the generic early features, retrain the task-specific late ones.
    """
    if args.finetune == "full":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    clf = list(model.get_classifier().parameters())
    clf_ids = {id(p) for p in clf}
    backbone = [(n, p) for n, p in model.named_parameters() if id(p) not in clf_ids]
    k = int(len(backbone) * args.freeze_frac)
    for _, p in backbone[:k]:
        p.requires_grad = False

    groups = [g for g in (
        {"params": [p for _, p in backbone[k:]], "lr": args.lr},
        {"params": clf, "lr": args.lr_head},
    ) if g["params"]]
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  partial fine-tune: froze {k}/{len(backbone)} backbone tensors -> "
          f"{n_tr/1e6:.2f}M / {n_all/1e6:.2f}M trainable "
          f"(backbone {args.lr:g} | head {args.lr_head:g})", flush=True)
    return torch.optim.AdamW(groups, weight_decay=args.wd)


def train_fold(train_rows, val_rows, args, device, fold):
    train_tf, eval_tf = RECIPES[args.recipe](args.img_size)
    tr = CropDataset(train_rows, train_tf)
    va = CropDataset(val_rows, eval_tf)
    gen = torch.Generator().manual_seed(args.seed * 1000 + fold)
    sampler = WeightedRandomSampler(class_weights(train_rows), len(train_rows),
                                    replacement=True, generator=gen)
    tl = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model = build_backbone(args.backbone, pretrained=not args.no_pretrained).to(device)
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

    model.eval()
    p, ytrue, subs = [], [], []
    with torch.no_grad():
        for x, y, s in vl:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1)[:, 1]
            if args.tta:
                # average P(anemic) over the image and its mirror (label-preserving)
                flip = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)[:, 1]
                prob = (prob + flip) / 2
            prob = prob.cpu().numpy()
            p.extend(prob.tolist()); ytrue.extend(y.tolist()); subs.extend(s)
    return subs, ytrue, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    help="timm model name, e.g. convnext_tiny | tf_efficientnetv2_s")
    ap.add_argument("--recipe", choices=list(RECIPES), default="colorhygiene_nowb",
                    help="locked Phase-2 recipe by default")
    ap.add_argument("--folds", default="data/folds.csv",
                    help="split file; e.g. data/folds_balanced_3.csv")
    ap.add_argument("--out", default=None,
                    help="default: runs/eval/<backbone>_<recipe>[_<split>]/seed<seed>")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="gentler than ResNet's 3e-4: ConvNeXt/EffNetV2 full fine-tune "
                         "is LR-sensitive and 3e-4 stalls folds at chance loss")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; keep 0 on Windows (spawn hangs)")
    ap.add_argument("--log-every", type=int, default=1,
                    help="print train loss every N epochs (0 = silent)")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average P(anemic) over image + hflip")
    ap.add_argument("--finetune", choices=["full", "partial"], default="full",
                    help="partial = freeze the earliest --freeze-frac of the backbone "
                         "and give the classifier its own LR")
    ap.add_argument("--freeze-frac", type=float, default=0.5,
                    help="partial mode: fraction of backbone tensors to freeze (forward order)")
    ap.add_argument("--lr-head", type=float, default=1e-3,
                    help="partial mode: classifier LR")
    ap.add_argument("--label-smoothing", type=float, default=None,
                    help="default: 0.05 for hygiene recipes, else 0.0")
    ap.add_argument("--cosine", dest="cosine", action="store_true", default=None)
    ap.add_argument("--no-cosine", dest="cosine", action="store_false")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    hygiene_opt = args.recipe in HYGIENE_OPT_RECIPES
    if args.label_smoothing is None:
        args.label_smoothing = 0.05 if hygiene_opt else 0.0
    if args.cosine is None:
        args.cosine = hygiene_opt
    split_tag = os.path.splitext(os.path.basename(args.folds))[0]
    suffix = "" if split_tag == "folds" else f"_{split_tag}"
    imgtag = "" if args.img_size == 224 else f"_img{args.img_size}"
    fttag = "" if args.finetune == "full" else f"_{args.finetune}"
    ttatag = "_tta" if args.tta else ""
    if args.out is None:
        args.out = (f"runs/eval/{args.backbone.replace('/', '_')}_{args.recipe}"
                    f"{fttag}{ttatag}{imgtag}{suffix}/seed{args.seed}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| backbone:", args.backbone, "| recipe:", args.recipe)

    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1

    oof = {}
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

    model_name = f"{args.backbone}_{args.recipe}"
    metrics = {
        "model": model_name,
        "oof": overall,
        "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                              for k in ("f1", "sensitivity", "specificity", "auroc", "auprc")},
        "per_fold": per_fold,
    }
    config = {"model": model_name, "backbone": args.backbone, "recipe": args.recipe,
              "folds": args.folds, "tta": args.tta, "finetune": args.finetune,
              "freeze_frac": args.freeze_frac, "lr_head": args.lr_head,
              "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "wd": args.wd,
              "label_smoothing": args.label_smoothing, "cosine_lr": args.cosine,
              "img_size": args.img_size, "seed": args.seed, "n_folds": n_folds,
              "pretrained": not args.no_pretrained,
              "selection": "fixed-epochs (no held-fold peeking)"}
    oof_rows = [[s, oof[s][0], round(oof[s][1], 6), oof[s][2]] for s in subjects]
    out = write_run(args.out, config, metrics, oof=oof_rows)

    print(f"\n=== {args.backbone} [{args.recipe}] (OOF over all {len(subjects)} subjects) ===")
    print(f"  AUROC       {overall['auroc']:.3f}   (colour floor = 0.890)")
    print(f"  sensitivity {overall['sensitivity']:.3f}   specificity {overall['specificity']:.3f}")
    print(f"  F1          {overall['f1']:.3f}   precision {overall['precision']:.3f}")
    print(f"  confusion   {overall['confusion']}")
    print(f"  per-fold F1 {ms('f1')[0]:.3f} +/- {ms('f1')[1]:.3f} | "
          f"AUROC {ms('auroc')[0]:.3f} +/- {ms('auroc')[1]:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
