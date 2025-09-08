#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, warnings, random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------
# Config
# -----------------------
DATA_ROOT = Path("/workspaces/ishi/datasets/anemia/raw/eyes_defy_anemia/dataset_anemia")
COUNTRIES = ["India", "Italy"]           # subfolders to scan
IMG_EXTS = (".jpg", ".jpeg", ".png")     # raw image extensions
MASK_PREFERENCE = ["forniceal_palpebral", "palpebral", "forniceal"]  # fallback order

IMAGE_SIZE = 256       # square resize for both image & mask
BATCH_SIZE = 8
EPOCHS     = 20
LR         = 1e-3
VAL_SPLIT  = 0.2
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR   = Path("models/crop")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BEST  = SAVE_DIR / "unet_crop_best.pth"

# -----------------------
# Repro
# -----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# -----------------------
# IO helpers
# -----------------------
def safe_open_image(path: Path, mode="RGB") -> Image.Image:
    """Robust image open (Pillow first, OpenCV fallback)."""
    # Pillow path
    try:
        with open(path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data))
        return img.convert(mode)
    except Exception:
        pass
    # OpenCV fallback
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cv2.imdecode failed: {path}")
    if mode == "RGBA":
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    else:  # RGB
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)

def find_mask_for_image(img_path: Path) -> Optional[Path]:
    """
    Given a raw image path .../XXXX.jpg, search for a mask with suffixes in MASK_PREFERENCE:
      XXXX_<maskkind>.png
    Return the first existing mask path, else None.
    """
    stem = img_path.stem  # '20200318_130148'
    parent = img_path.parent
    for kind in MASK_PREFERENCE:
        # Support Italy patterns that sometimes contain extra underscores or T_* stems
        # We primarily rely on '<stem>_<kind>.png'
        candidate = parent / f"{stem}_{kind}.png"
        if candidate.exists():
            return candidate

        # Also try variants where original images are like "1.jpg" and masks named "001_palpebral.png"
        # We won't overfit; but if a mask shares a prefix and kind, take it.
        # (Only used as last resort)
        for p in parent.glob(f"*{kind}*.png"):
            # favor exact prefix match if present
            if stem in p.stem:
                return p
    return None

def rgba_to_binary_mask(mask_img: Image.Image) -> np.ndarray:
    """
    Convert RGBA (or RGB/Gray) mask to a binary numpy mask [H, W] {0,1}.
    Prefer alpha channel; otherwise >0 anywhere.
    """
    arr = np.array(mask_img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        binm = (arr[:, :, 3] > 0).astype(np.float32)
    elif arr.ndim == 3:
        binm = (arr.mean(axis=2) > 0).astype(np.float32)
    else:
        binm = (arr > 0).astype(np.float32)
    return binm

# -----------------------
# Dataset
# -----------------------
class EyelidSegDataset(Dataset):
    def __init__(self, roots: List[Path], img_size: int = 256, augment: bool = True):
        self.items: List[Tuple[Path, Path]] = []
        self.size = img_size
        self.augment = augment

        # collect (image, mask) pairs
        for root in roots:
            if not root.exists():
                continue
            for img_path in root.rglob("*"):
                if img_path.suffix.lower() in IMG_EXTS:
                    mpath = find_mask_for_image(img_path)
                    if mpath is None:
                        continue
                    self.items.append((img_path, mpath))

        if len(self.items) == 0:
            raise RuntimeError("No (image, mask) pairs found. Check DATA_ROOT and folder layout.")

        print(f"Found {len(self.items)} pairs.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ipath, mpath = self.items[idx]

        # robust open
        try:
            img = safe_open_image(ipath, mode="RGB")
        except Exception as e:
            warnings.warn(f"[SKIP] bad image: {ipath} ({e})")
            return self.__getitem__((idx + 1) % len(self))

        try:
            m = safe_open_image(mpath, mode="RGBA")
        except Exception as e:
            warnings.warn(f"[SKIP] bad mask: {mpath} ({e})")
            return self.__getitem__((idx + 1) % len(self))

        # resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        m   = m.resize((self.size, self.size), Image.NEAREST)

        # augment (very light)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            m   = m.transpose(Image.FLIP_LEFT_RIGHT)

        # to tensors
        img = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
        img = np.transpose(img, (2, 0, 1))                # [3,H,W]
        mask = rgba_to_binary_mask(m)                     # [H,W] {0,1}

        img_t  = torch.from_numpy(img)                    # [3,H,W]
        mask_t = torch.from_numpy(mask).unsqueeze(0)      # [1,H,W]

        return img_t, mask_t

# -----------------------
# UNet (small)
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)        # 3->32
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)       # 32->64
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)     # 64->128
        self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*4, base*8)     # 128->256

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.u_d3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.u_d2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.u_d1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        x  = self.p1(c1)
        c2 = self.d2(x)
        x  = self.p2(c2)
        c3 = self.d3(x)
        x  = self.p3(c3)
        x  = self.b(x)
        x  = self.u3(x)
        x  = torch.cat([x, c3], dim=1)
        x  = self.u_d3(x)
        x  = self.u2(x)
        x  = torch.cat([x, c2], dim=1)
        x  = self.u_d2(x)
        x  = self.u1(x)
        x  = torch.cat([x, c1], dim=1)
        x  = self.u_d1(x)
        return self.out(x)   # logits

# -----------------------
# Loss & metrics
# -----------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    # pred, target in [0,1], shape [B,1,H,W]
    num = (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return (2 * num / den).mean()

def iou_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - inter + eps
    return (inter / union).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_coeff(probs, target)
        return self.w * bce + (1 - self.w) * dice

# -----------------------
# Train / Val loops
# -----------------------
def build_datasets() -> Tuple[Dataset, Dataset]:
    roots = [DATA_ROOT / c for c in COUNTRIES]
    ds = EyelidSegDataset(roots, img_size=IMAGE_SIZE, augment=True)
    val_len = max(1, int(VAL_SPLIT * len(ds)))
    train_len = len(ds) - val_len
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)
    return train_ds, val_ds

def run():
    print(f"Device: {DEVICE}")
    train_ds, val_ds = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          num_workers=2, persistent_workers=True)


    model = UNetSmall(in_ch=3, out_ch=1, base=32).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = BCEDiceLoss(0.5)

    best_val = -1.0

    for epoch in range(1, EPOCHS+1):
        # ---- Train
        model.train()
        tr_loss, tr_dice, tr_iou = 0.0, 0.0, 0.0
        for img, m in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            img = img.to(DEVICE, non_blocking=True)
            m   = m.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(img)
            loss = crit(logits, m)
            loss.backward()
            opt.step()

            probs = torch.sigmoid(logits).clamp_(0,1)
            tr_loss += loss.item()
            tr_dice += dice_coeff(probs, m).item()
            tr_iou  += iou_coeff(probs, m).item()

        ntr = len(train_loader)
        print(f"Train | loss {tr_loss/ntr:.4f} dice {tr_dice/ntr:.4f} iou {tr_iou/ntr:.4f}")

        # ---- Val
        model.eval()
        v_loss, v_dice, v_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for img, m in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                img = img.to(DEVICE, non_blocking=True)
                m   = m.to(DEVICE, non_blocking=True)
                logits = model(img)
                loss = crit(logits, m)
                probs = torch.sigmoid(logits).clamp_(0,1)
                v_loss += loss.item()
                v_dice += dice_coeff(probs, m).item()
                v_iou  += iou_coeff(probs, m).item()

        nval = len(val_loader)
        v_loss /= nval
        v_dice /= nval
        v_iou  /= nval
        print(f"Val   | loss {v_loss:.4f} dice {v_dice:.4f} iou {v_iou:.4f}")

        # Save best by Dice
        if v_dice > best_val:
            best_val = v_dice
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": v_dice,
                        "val_iou": v_iou,
                        "img_size": IMAGE_SIZE},
                       CKPT_BEST)
            print(f"✅ Saved best model (dice={v_dice:.4f}) -> {CKPT_BEST}")

    print("Done.")

if __name__ == "__main__":
    run()
