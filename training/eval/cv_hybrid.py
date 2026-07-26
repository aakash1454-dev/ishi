#!/usr/bin/env python3
"""
HYBRID: ResNet18 image features + the 20 hand-built colour features, fused in the head.

Rationale. The colour model (20 measurements -> logistic regression) beats every deep
model we tried, most starkly inside India (0.722 vs 0.611). The CNN clearly struggles
to rediscover "average redness" from 217 examples. So instead of hoping it learns that,
we hand it the colour numbers directly and let it use them alongside whatever it can
learn from the pixels.

    image --> ResNet18 --> 512 features --.
                                           +--> [532] --> Linear --> anemic / not
    image --> 20 colour measurements -----'

The colour features are standardised using TRAIN-FOLD statistics only, so no
information leaks from the held-out fold.

Note the counter-evidence: colour stacked with FROZEN DINOv2 features made things
worse (0.890 -> 0.847) -- the deep features diluted the colour signal. This run tests
whether joint TRAINING (rather than post-hoc stacking) behaves differently.

Run:
  uv run python -m training.eval.cv_hybrid --folds data/folds_balanced.csv --epochs 20
"""
import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from training.eval.dataset import RECIPES, load_folds, class_weights, CROPS
from training.eval.baseline_color import features as colour_features
from training.eval.runlog import screening_metrics, write_run


def colour_table(rows, crops=CROPS):
    """subject -> 20-dim colour descriptor (deterministic, computed once)."""
    out = {}
    for r in rows:
        out[r["subject"]] = colour_features(pathlib.Path(crops) / r["image"])
    return out


class HybridDataset(Dataset):
    """Returns (image_tensor, colour_vector, y, subject)."""
    def __init__(self, rows, transform, table, mu, sd, crops=CROPS):
        self.rows, self.tf, self.table = rows, transform, table
        self.mu, self.sd = mu, sd
        self.crops = pathlib.Path(crops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.crops / r["image"]).convert("RGB")
        c = (self.table[r["subject"]] - self.mu) / self.sd
        return self.tf(img), torch.tensor(c, dtype=torch.float32), r["y"], r["subject"]


class HybridNet(nn.Module):
    def __init__(self, n_colour=20, pretrained=True, dropout=0.0):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        d = self.feat_dim + n_colour
        head = [nn.Dropout(dropout)] if dropout > 0 else []
        head.append(nn.Linear(d, 2))
        self.head = nn.Sequential(*head)

    def forward(self, x, c):
        return self.head(torch.cat([self.backbone(x), c], dim=1))


def build_optimizer(model, args):
    """Same partial fine-tune scheme as the classifier (+0.042), adjusted for the
    'backbone.' prefix introduced by the wrapper module."""
    if args.finetune == "full":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    frozen = tuple(f"backbone.{p}" for p in ("conv1", "bn1", "layer1", "layer2"))
    for n, p in model.named_parameters():
        if n.startswith(frozen):
            p.requires_grad = False
    sel = lambda pre: [p for n, p in model.named_parameters()
                       if n.startswith(pre) and p.requires_grad]
    groups = [g for g in (
        {"params": sel("backbone.layer3"), "lr": args.lr_layer3},
        {"params": sel("backbone.layer4"), "lr": args.lr_layer4},
        {"params": sel("head"),            "lr": args.lr_head},
    ) if g["params"]]
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"  hybrid partial fine-tune: {n_tr/1e6:.2f}M trainable", flush=True)
    return torch.optim.AdamW(groups, weight_decay=args.wd)


def train_fold(train_rows, val_rows, table, args, device, fold):
    train_tf, eval_tf = RECIPES[args.recipe](args.img_size)
    # standardise colour features on the TRAIN fold only
    tr_c = np.stack([table[r["subject"]] for r in train_rows])
    mu, sd = tr_c.mean(0), tr_c.std(0) + 1e-6

    tr = HybridDataset(train_rows, train_tf, table, mu, sd)
    va = HybridDataset(val_rows, eval_tf, table, mu, sd)
    gen = torch.Generator().manual_seed(args.seed * 1000 + fold)
    sampler = WeightedRandomSampler(class_weights(train_rows), len(train_rows),
                                    replacement=True, generator=gen)
    tl = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model = HybridNet(n_colour=tr_c.shape[1], pretrained=not args.no_pretrained,
                      dropout=args.dropout).to(device)
    opt = build_optimizer(model, args)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for ep in range(args.epochs):
        model.train(); tot = nb = 0
        for x, c, y, _ in tl:
            x, c, y = x.to(device), c.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x, c), y)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if args.log_every and (ep + 1) % args.log_every == 0:
            print(f"    epoch {ep+1:2d}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    model.eval(); p, ytrue, subs = [], [], []
    with torch.no_grad():
        for x, c, y, s in vl:
            x, c = x.to(device), c.to(device)
            prob = torch.softmax(model(x, c), dim=1)[:, 1]
            if args.tta:      # colour features are flip-invariant, so only the image flips
                prob = (prob + torch.softmax(model(torch.flip(x, dims=[3]), c), dim=1)[:, 1]) / 2
            p.extend(prob.cpu().numpy().tolist()); ytrue.extend(y.tolist()); subs.extend(s)
    return subs, ytrue, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES), default="legacy")
    ap.add_argument("--folds", default="data/folds_balanced.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune", choices=["full", "partial"], default="partial")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-4)
    ap.add_argument("--lr-layer3", type=float, default=1e-5)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"runs/eval/hybrid_resnet18_colour{'_tta' if args.tta else ''}/seed{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_folds(args.folds)
    n_folds = max(r["fold"] for r in rows) + 1
    print(f"device: {device} | computing colour features for {len(rows)} crops ...")
    table = colour_table(rows)
    print(f"  colour vector length: {len(next(iter(table.values())))}")

    oof, per_fold = {}, []
    for f in range(n_folds):
        t0 = time.time()
        subs, y, p = train_fold([r for r in rows if r["fold"] != f],
                                [r for r in rows if r["fold"] == f], table, args, device, f)
        for s, yy, pp in zip(subs, y, p):
            oof[s] = (yy, pp, f)
        m = screening_metrics(y, p); per_fold.append(m)
        print(f"  fold {f}: AUROC {m['auroc']:.3f} F1 {m['f1']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    subs = sorted(oof)
    y = [oof[s][0] for s in subs]; p = [oof[s][1] for s in subs]
    overall = screening_metrics(y, p)
    ms = lambda k: (float(np.mean([m[k] for m in per_fold])),
                    float(np.std([m[k] for m in per_fold])))
    metrics = {"model": "hybrid_resnet18_colour", "oof": overall,
               "per_fold_mean_std": {k: {"mean": ms(k)[0], "std": ms(k)[1]}
                                     for k in ("f1", "sensitivity", "specificity", "auroc")},
               "per_fold": per_fold}
    config = {"model": "hybrid_resnet18_colour", "recipe": args.recipe, "folds": args.folds,
              "finetune": args.finetune, "tta": args.tta, "epochs": args.epochs,
              "seed": args.seed, "fusion": "concat(resnet512, colour20) -> linear"}
    out = write_run(args.out, config,
                    metrics, oof=[[s, oof[s][0], round(oof[s][1], 6), oof[s][2]] for s in subs])
    print(f"\n=== HYBRID (ResNet18 + colour features) ===")
    print(f"  AUROC {overall['auroc']:.3f} | F1 {overall['f1']:.3f} | "
          f"sens {overall['sensitivity']:.3f} | spec {overall['specificity']:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
