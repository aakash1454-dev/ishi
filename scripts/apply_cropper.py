#!/usr/bin/env python3
"""
Apply the trained UNetSmall conjunctiva cropper to a folder of images, writing a
binary mask (255=conjunctiva) per image at the image's native size.

Used to rescue the 31 Italy(1-31) subjects (their prepped photos are already
EXIF-corrected, so omit --exif), and reusable as the production cropper.

Run:
  uv run python scripts/apply_cropper.py \
      --images data/rescue_italy_1_31/images \
      --masks  data/rescue_italy_1_31/masks
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from training.crop.train_unet import UNetSmall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/crop/unet_crop_best.pth")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--exif", action="store_true", help="EXIF-transpose inputs (raw jpgs)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    size = ck.get("img_size", 256)
    model = UNetSmall().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded cropper (val_dice={ck.get('val_dice','?')}, img_size={size}) on {dev}")

    src = pathlib.Path(args.images)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    out = pathlib.Path(args.masks); out.mkdir(parents=True, exist_ok=True)

    for p in files:
        im = Image.open(p)
        if args.exif:
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB"); W, H = im.size
        x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)))[None].to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = (pr > args.thr).astype(np.uint8) * 255
        Image.fromarray(mask).resize((W, H), Image.NEAREST).save(out / f"{p.stem}.png")

    print(f"wrote {len(files)} masks -> {out}")


if __name__ == "__main__":
    main()
