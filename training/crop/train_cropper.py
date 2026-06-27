#!/usr/bin/env python3
"""
Train a UNetSmall conjunctiva cropper on the 186 cleanly hand-segmented Eyes-Defy
subjects, then it can be applied (scripts/apply_cropper.py) to rescue the 31 Italy
subjects (1-31) whose hand-masks are degenerate.

Reuses the architecture/loss/metrics from training/crop/train_unet.py. Differences
that matter here:
  - EXIF-corrects each original .jpg so it aligns with the (portrait) hand-mask
    before resizing  (train_unet.py skipped this -> image/mask were misaligned).
  - Filters masks by alpha coverage (0.5%..90%) so degenerate full-frame masks
    (Italy 1-31) and empty/corrupt ones are excluded from training.

Run (GPU):
  uv run python -m training.crop.train_cropper --epochs 40
"""
import argparse
import pathlib
import random
import sys

import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from training.crop.train_unet import UNetSmall, BCEDiceLoss, dice_coeff, iou_coeff

RAW = pathlib.Path("datasets/anemia/raw/eyes_defy_anemia/dataset_anemia")
IMG_SIZE = 256
MASK_PREFERENCE = ("forniceal_palpebral", "palpebral")  # fuller conjunctiva region first


def best_mask(folder: pathlib.Path):
    for kind in MASK_PREFERENCE:
        for m in folder.glob(f"*_{kind}.png"):
            return m
    return None


def build_pairs():
    """Return [(subject_key, jpg_path, mask_path)] for usable subjects only."""
    pairs = []
    for country in ("India", "Italy"):
        for d in (RAW / country).iterdir():
            if not d.is_dir():
                continue
            jpg = list(d.glob("*.jpg"))
            m = best_mask(d)
            if not jpg or m is None:
                continue
            try:
                cov = (np.array(Image.open(m).convert("RGBA"))[..., 3] > 0).mean()
            except Exception:
                continue
            if 0.005 < cov < 0.90:  # drop degenerate full-frame, empty, corrupt
                pairs.append((f"{country}_{d.name}", jpg[0], m))
    return pairs


class CropDataset(Dataset):
    def __init__(self, pairs, size=IMG_SIZE, augment=True):
        self.pairs, self.size, self.augment = pairs, size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        _, jpg, mpath = self.pairs[i]
        img = ImageOps.exif_transpose(Image.open(jpg)).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        m = Image.open(mpath).convert("RGBA").resize((self.size, self.size), Image.NEAREST)
        mask = (np.array(m)[..., 3] > 0).astype(np.float32)
        img = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:        # horizontal flip (geometry only)
            img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy()
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/crop/unet_crop_best.pth")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    pairs = build_pairs()
    random.shuffle(pairs)
    nval = max(1, int(args.val_frac * len(pairs)))
    val, train = pairs[:nval], pairs[nval:]
    print(f"usable pairs: {len(pairs)} (train {len(train)}, val {len(val)})")

    tl = DataLoader(CropDataset(train, augment=True), batch_size=args.bs, shuffle=True, num_workers=0)
    vl = DataLoader(CropDataset(val, augment=False), batch_size=args.bs, num_workers=0)

    model = UNetSmall().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = BCEDiceLoss(0.5)
    best = -1.0
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        for img, m in tl:
            img, m = img.to(dev), m.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(img), m)
            loss.backward(); opt.step()

        model.eval(); vd = vi = 0.0; n = 0
        with torch.no_grad():
            for img, m in vl:
                img, m = img.to(dev), m.to(dev)
                p = torch.sigmoid(model(img))
                vd += dice_coeff(p, m).item(); vi += iou_coeff(p, m).item(); n += 1
        vd /= n; vi /= n
        print(f"epoch {ep:02d}/{args.epochs}  val_dice {vd:.4f}  val_iou {vi:.4f}")
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "img_size": IMG_SIZE,
                        "val_dice": vd, "arch": "UNetSmall"}, args.out)

    print(f"best val_dice {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
