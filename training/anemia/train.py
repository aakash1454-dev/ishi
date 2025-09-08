#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt


# ------------------------
# Helpers / Utilities
# ------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def split_indices(n, val_ratio, seed):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def make_weighted_sampler(imagefolder, indices):
    # imagefolder must be a torchvision.datasets.ImageFolder (has .samples)
    counts = {}
    for i in indices:
        y = imagefolder.samples[i][1]
        counts[y] = counts.get(y, 0) + 1
    class_weights = {k: 1.0 / float(v) for k, v in counts.items()}
    sample_weights = [class_weights[imagefolder.samples[i][1]] for i in indices]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def unwrap_to_imagefolder_and_indices(ds):
    """
    Returns (base_imagefolder, indices_into_base) for an arbitrarily nested Subset(ImageFolder).
    """
    if isinstance(ds, Subset):
        base, base_indices = unwrap_to_imagefolder_and_indices(ds.dataset)
        eff = [base_indices[i] for i in ds.indices]
        return base, eff
    # base case: ImageFolder
    if not hasattr(ds, "samples"):
        raise TypeError("Expected ImageFolder or Subset(ImageFolder)")
    return ds, list(range(len(ds.samples)))


def build_model(num_classes, backbone="resnet18", pretrained=True):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return m


def plot_loss_curve(train_losses, val_losses, out_png):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_confusion(cm, class_names, out_png):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm


def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-classes", type=str, default="data/anemia_crops",
                    help="Root folder. If it has train/ and val/, those are used. Otherwise expects class subdirs (anemic/, nonanemic/ or non_anemic/).")
    ap.add_argument("--out", type=str, default="runs/anemia_resnet18_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--subset", type=int, default=0, help="Train on first N samples after split (0=all)")
    ap.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v3_small"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")

    # eval mode
    ap.add_argument("--eval-image", type=str, default="", help="Path to single image to classify")
    ap.add_argument("--checkpoint", type=str, default="", help="Model checkpoint for eval")

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf, val_tf = build_transforms(args.img_size)

    # --- Dataset loading ---
    # Prefer pre-split folders: root/train/* and root/val/*; ignore any class not in allowed (e.g., 'unknown')
    root = Path(args.root_classes)
    train_root = root / "train"
    val_root   = root / "val"

    # Allow common class-name variants; canonicalize to ['anemic','nonanemic']
    synonym_map = {
        "anemic": {"anemic"},
        "nonanemic": {"nonanemic", "non_anemic", "non-anemic", "normal", "not_anemic"},
    }

    def canonicalize_present(class_list):
        present = set(c.lower() for c in class_list)
        canon = []
        for canon_name, variants in synonym_map.items():
            if present & variants:
                canon.append(canon_name)
        return sorted(canon)

    def load_split_dir(root_dir, transform):
        ds = datasets.ImageFolder(str(root_dir), transform=transform)
        present_canon = canonicalize_present(ds.classes)
        # Build concrete class names to allow (based on what is present)
        allow_concrete = set()
        for canon_name in present_canon:
            allow_concrete |= synonym_map[canon_name]
        # Filter to allowed classes only
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        keep = [i for i, (_, ci) in enumerate(ds.samples) if idx_to_class[ci].lower() in allow_concrete]
        if not keep:
            raise RuntimeError(f"No images found in {root_dir} for allowed classes {allow_concrete}.")
        return Subset(ds, keep), present_canon

    use_pre_split = False
    if train_root.is_dir() and val_root.is_dir():
        # Pre-split path
        train_ds, train_present = load_split_dir(train_root, train_tf)
        val_ds,   val_present   = load_split_dir(val_root,   val_tf)
        if set(train_present) != set(val_present):
            raise RuntimeError(f"Train/val classes mismatch after mapping: {train_present} vs {val_present}")
        # Force canonical order
        class_names = ["anemic", "nonanemic"]
        print("Classes:", class_names)
        use_pre_split = True

        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

    else:
        # Fallback: expect root has class subdirs -> do random split here
        # Build a train-time ImageFolder (with train transforms)
        ds_train = datasets.ImageFolder(str(root), transform=train_tf)
        present = canonicalize_present(ds_train.classes)
        if set(present) != {"anemic", "nonanemic"}:
            print("Classes detected under", root, "->", ds_train.classes, "mapped to", present)
        if len(present) < 2:
            raise RuntimeError(f"Expected both 'anemic' and a 'non-anemic' variant under {root}. Found: {ds_train.classes}")

        # Allow concrete names found
        allow_concrete = set()
        for canon_name in present:
            allow_concrete |= synonym_map[canon_name]

        idx_to_class = {v: k for k, v in ds_train.class_to_idx.items()}
        keep = [i for i, (_, ci) in enumerate(ds_train.samples) if idx_to_class[ci].lower() in allow_concrete]
        full_ds = Subset(ds_train, keep)

        class_names = ["anemic", "nonanemic"]
        print("Classes:", class_names)

        n = len(full_ds)
        train_idx, val_idx = split_indices(n, args.val_ratio, args.seed)
        if args.subset and args.subset > 0:
            train_idx = train_idx[:min(args.subset, len(train_idx))]
            print(f"Using subset for training: {len(train_idx)} samples")
        else:
            print(f"Training samples: {len(train_idx)}")

        # Train subset (with train transforms already)
        train_ds = Subset(full_ds, train_idx)

        # Validation dataset should use val transforms; rebuild base with val_tf
        ds_val = datasets.ImageFolder(str(root), transform=val_tf)
        idx_to_class_v = {v: k for k, v in ds_val.class_to_idx.items()}
        keep_v = [i for i, (_, ci) in enumerate(ds_val.samples) if idx_to_class_v[ci].lower() in allow_concrete]
        val_base = Subset(ds_val, keep_v)
        val_ds = Subset(val_base, val_idx)

        # Weighted sampler for imbalance — robust to nested Subset(ImageFolder)
        base_imgfolder, eff_indices = unwrap_to_imagefolder_and_indices(train_ds)
        sampler = make_weighted_sampler(base_imgfolder, eff_indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.bs,
            sampler=sampler,
            shuffle=False,          # don't shuffle when using a sampler
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

    # --- Model / Optim / AMP ---
    model = build_model(num_classes=2, backbone=args.backbone, pretrained=not args.no_pretrained)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # --- Train ---
    best_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, device, optimizer, scaler)
        train_losses.append(train_loss)

        # validation pass
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        val_total = 0.0
        val_n = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_total += float(loss.item()) * x.size(0)
                val_n += x.size(0)
                pred = torch.argmax(logits, dim=1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        val_loss = val_total / max(val_n, 1)
        val_losses.append(val_loss)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(np.array(y_true), np.array(y_pred), average="binary")
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  "
              f"time={time.time()-t0:.1f}s")

        # save last
        torch.save({"model": model.state_dict(),
                    "classes": class_names,
                    "backbone": args.backbone},
                   str(out_dir / "last.pth"))

        # save best by val acc
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(),
                        "classes": class_names,
                        "backbone": args.backbone},
                       str(out_dir / "best_acc.pth"))

        # update loss curve each epoch
        plot_loss_curve(train_losses, val_losses, str(out_dir / "loss_curve.png"))

    # final confusion matrix on the last val predictions
    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    plot_confusion(cm, class_names, str(out_dir / "confusion_matrix.png"))
    print("Best val acc:", f"{best_acc:.4f}")

    # --- Optional: single-image eval ---
    if args.eval_image:
        ckpt_path = args.checkpoint if args.checkpoint else str(out_dir / "best_acc.pth")
        if not os.path.isfile(ckpt_path):
            print("Checkpoint not found for eval:", ckpt_path)
            return
        ckpt = torch.load(ckpt_path, map_location=device)
        if "backbone" in ckpt and ckpt["backbone"] != args.backbone:
            print("Warning: backbone mismatch between checkpoint and args.")

        model = build_model(num_classes=2, backbone=args.backbone, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        model.eval()

        infer_tf = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        from PIL import Image
        img = Image.open(args.eval_image).convert("RGB")
        x = infer_tf(img).unsqueeze(0).to(device)
        THRESHOLD = 0.60
        p_anemic = float(probs[class_names.index("anemic")])
        label = "anemic" if p_anemic >= THRESHOLD else "nonanemic"
        print("Classes:", class_names)
        print("Prediction:", label)
        print("p_anemic:", p_anemic, "Threshold:", THRESHOLD)
        print("Raw probs:", probs.tolist())



if __name__ == "__main__":
    main()
