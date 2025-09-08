# training/anemia/train_mobilenetv3.py
# Minimal, readable training loop with MobileNetV3 + RandomErasing, balanced sampler, and checkpointing.

import os, argparse, time, random
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_datasets(data_root, img_size, subset):
    # Expecting ImageFolder with class subdirs: nonanemic/ anemic/
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)

    if subset > 0 and subset < len(train_ds):
        idx = np.random.choice(len(train_ds), subset, replace=False)
        train_ds.samples = [train_ds.samples[i] for i in idx]
        train_ds.targets = [train_ds.targets[i] for i in idx]

    return train_ds, val_ds

def make_balanced_sampler(targets):
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def build_model(arch, pretrained):
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, 1)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, 1)
    else:
        raise ValueError("arch must be mobilenet_v3_small or mobilenet_v3_large")
    return m

def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            logits = out.view(-1) if out.shape[-1] == 1 else out[:,1]
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    # Default threshold for metrics during training; final tuning later.
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    pred = (probs >= 0.75).astype(np.int64)
    cm = confusion_matrix(all_targets, pred, labels=[0,1])  # 0=non-anemic, 1=anemic
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(tn+fp+fn+tp, 1)
    try:
        auc = roc_auc_score(all_targets, probs)
    except:
        auc = float("nan")
    return acc, auc, tn, fp, fn, tp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with train/ and val/ folders")
    ap.add_argument("--arch", default="mobilenet_v3_small")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="use N samples of train for fast iteration; 0=all")
    ap.add_argument("--out", default="models/anemia/mobilenetv3_small_randomerase.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = build_datasets(args.data, args.img_size, args.subset)
    sampler = make_balanced_sampler(train_ds.targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)

    model = build_model(args.arch, args.pretrained).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_auc = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)  # BCE expects float targets
            y = y.view(-1, 1)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = run_loss / max(n, 1)
        acc, auc, tn, fp, fn, tp = evaluate(model, val_loader, device)

        print("Epoch %d | train_loss=%.4f | val_acc=%.4f | val_auc=%.4f | TN=%d FP=%d FN=%d TP=%d" %
              (epoch, train_loss, acc, auc, tn, fp, fn, tp))

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.out)
            print("Saved best model -> %s" % args.out)

    print("Training complete. Best AUC = %.4f" % best_auc)

if __name__ == "__main__":
    main()
