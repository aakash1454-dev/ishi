#!/usr/bin/env python3
import os, csv, json, random
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class CropDataset(Dataset):
    def __init__(self, manifest_csv: Path, aug: bool):
        self.rows = []
        with open(manifest_csv, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append({"image": row["image"], "label": int(row["label"])})
        if aug:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        p = self.rows[idx]["image"]
        y = self.rows[idx]["label"]
        img = Image.open(p).convert("RGB")
        return self.tf(img), torch.tensor([y], dtype=torch.float32)

def make_sampler(labels):
    # balance classes in each epoch
    class_counts = np.bincount(labels, minlength=2)
    inv = 1.0 / np.maximum(class_counts, 1)
    weights = [inv[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.extend(y.cpu().numpy().reshape(-1).tolist())
            ps.extend(prob.tolist())
    ys = np.array(ys, dtype=np.int32)
    ps = np.array(ps, dtype=np.float32)
    # search best F1 threshold on this split
    ts = np.linspace(0,1,501)
    best_t, best_f1 = 0.5, 0.0
    for t in ts:
        pred = (ps >= t).astype(np.int32)
        f1 = f1_score(ys, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (ps >= best_t).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(ys, pred, average="binary", zero_division=0)
    cm = confusion_matrix(ys, pred, labels=[0,1])
    return dict(
        best_threshold=float(best_t),
        f1=float(f1),
        precision=float(p),
        recall=float(r),
        cm=cm.tolist(),
        y_true=ys.tolist(),
        y_prob=ps.tolist()
    )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-root", default="/workspaces/ishi/cached_crops")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/workspaces/ishi/models/anemia/anemia_cnn_ft.pth")
    args = ap.parse_args()

    set_seed(1337)
    root = Path(args.crops_root)
    train_csv = root / "train" / "manifest.csv"
    val_csv   = root / "val" / "manifest.csv"
    assert train_csv.exists() and val_csv.exists(), "Run cache_crops.py first."

    # datasets
    dtr = CropDataset(train_csv, aug=True)
    dva = CropDataset(val_csv,   aug=False)

    # loaders (balanced sampler for train)
    train_labels = [r["label"] for r in dtr.rows]
    sampler = make_sampler(train_labels)
    tr_loader = DataLoader(dtr, batch_size=args.bs, sampler=sampler, num_workers=2, pin_memory=True)
    va_loader = DataLoader(dva, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.to(device)

    # loss with positive weighting
    pos = max(sum(train_labels), 1)
    neg = max(len(train_labels) - pos, 1)
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1
    best_state = None
    for ep in range(1, args.epochs+1):
        net.train()
        running = 0.0
        for x, y in tr_loader:
            x = x.to(device); y = y.to(device)
            logits = net(x).squeeze(1)
            loss = criterion(logits, y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running / len(dtr)

        # val
        val_metrics = evaluate(net, va_loader, device)
        print(f"[ep {ep}] loss={tr_loss:.4f}  val_f1={val_metrics['f1']:.4f}  "
              f"thr={val_metrics['best_threshold']:.3f}  "
              f"prec={val_metrics['precision']:.3f}  rec={val_metrics['recall']:.3f}  "
              f"cm={val_metrics['cm']}")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "model": net.state_dict(),
                "val": val_metrics,
                "pos_weight": float(pos_weight.item()),
                "normalize_mean": [0.485,0.456,0.406],
                "normalize_std":  [0.229,0.224,0.225],
                "arch": "resnet18"
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    info_json = Path(args.out).with_suffix(".json")
    with open(info_json, "w") as f:
        json.dump(best_state["val"], f, indent=2)
    print(f"Saved fine-tuned model -> {args.out}")
    print(f"Val summary -> {info_json}")

if __name__ == "__main__":
    main()
