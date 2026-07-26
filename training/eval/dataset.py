#!/usr/bin/env python3
"""
Shared dataset + transforms for the eval harness. Reads data/folds.csv (the one
trusted subject-level split) and serves crops from data/crops/.

`legacy` transforms reproduce the CURRENT model's recipe (ColorJitter incl.
saturation/hue) so the ResNet18 re-baseline measures the existing approach
honestly. The Phase-2 "color-hygiene" recipe will be a separate transform set.
"""
import csv
import pathlib

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CROPS = pathlib.Path("data/crops")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_folds(path="data/folds.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["y"] = int(r["y"]); r["fold"] = int(r["fold"])
    return rows


def legacy_transforms(img_size=224):
    """The current production recipe (kept verbatim for an honest re-baseline)."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class GrayWorld:
    """Gray-world color constancy. Rescales each RGB channel so its mean equals the
    image's overall mean brightness, neutralizing per-camera/lighting color casts.

    This is a PREPROCESSING step (applied to train AND eval): the anemia cue is the
    conjunctiva's color, so capture-condition color casts must be removed rather than
    learned. Operates on a PIL RGB image, returns a PIL RGB image.
    """
    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32)          # H,W,3 in [0,255]
        means = arr.reshape(-1, 3).mean(axis=0)          # per-channel mean
        gray = float(means.mean())                       # target neutral level
        scale = gray / np.clip(means, 1e-6, None)        # per-channel gain
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def colorhygiene_transforms(img_size=224):
    """Phase-2 'color-hygiene' recipe (the plan's #1 fix).

    vs legacy: gray-world white balance (train+eval); NO saturation/hue jitter (only a
    tiny brightness wiggle for exposure); richer label-preserving geometry (rotation,
    translation, mild zoom via RandomResizedCrop) + random erasing. Nothing here alters
    pallor, so labels stay valid.
    """
    train_tf = transforms.Compose([
        GrayWorld(),
        transforms.RandomAffine(degrees=12, translate=(0.06, 0.06)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1),          # tiny exposure only; no sat/hue
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    eval_tf = transforms.Compose([
        GrayWorld(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def legacy_nosathue_transforms(img_size=224):
    """Ablation: legacy recipe with ONLY saturation/hue removed from ColorJitter.

    The surgical test of the plan's core hypothesis ('sat/hue jitter scrambles the
    colour cue'). Everything else is legacy, so a gain here vs legacy is attributable
    to that one change alone. Pair with legacy optimization (no label smoothing/cosine).
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # dropped saturation+hue
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def colorhygiene_nowb_transforms(img_size=224):
    """Ablation: the full colorhygiene recipe MINUS gray-world white balance.

    Compared against `colorhygiene` (run with the same label-smoothing + cosine), the
    ONLY difference is gray-world, so the gap isolates whether WB on tight conjunctiva
    crops is destroying the colour signal (the prime suspect for the AUROC drop).
    """
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.06, 0.06)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def domainrand_transforms(img_size=224):
    """Domain randomization aimed at CROSS-POPULATION generalization.

    The confound is capture conditions (camera + lighting) baked into the pixels, and
    the deployment target is 'any new population'. Rather than remove colour variation
    (gray-world / CLAHE both failed), we AMPLIFY it during training so the model learns
    to read pallor despite many simulated cameras/lighting, instead of keying on one
    site's look. Judge on the leave-one-country-out score, NOT the pooled score.

    We do NOT normalise colour and we DO push brightness/contrast/white-balance jitter
    hard, plus light blur + erasing to mimic focus/sensor differences. Pallor-critical
    saturation/hue are jittered only mildly so the signal survives.
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


RECIPES = {
    "legacy": legacy_transforms,
    "colorhygiene": colorhygiene_transforms,
    "legacy_nosathue": legacy_nosathue_transforms,
    "colorhygiene_nowb": colorhygiene_nowb_transforms,
    "domainrand": domainrand_transforms,
}

# Recipes that carry the colorhygiene optimization bundle (label smoothing + cosine LR)
# by default, so an ablation differs from `colorhygiene` only in its transforms.
HYGIENE_OPT_RECIPES = {"colorhygiene", "colorhygiene_nowb"}


class CropDataset(Dataset):
    """RECTANGLE crops (data/crops/): bbox around the conjunctiva, skin included.
    rows: list of folds.csv dicts. Returns (image_tensor, y, subject)."""
    def __init__(self, rows, transform, crops=CROPS):
        self.rows = rows
        self.tf = transform
        self.crops = pathlib.Path(crops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.crops / r["image"]).convert("RGB")
        return self.tf(img), r["y"], r["subject"]


MASKED_ROOT = pathlib.Path("datasets/eda-masked-crops")


def masked_path_map(root=MASKED_ROOT):
    """subject key ('India_1') -> path of that subject's masked RGBA crop.

    The masked dataset ids subjects as 'eda_india_1'; folds.csv uses 'India_1',
    so we translate to keep the rectangle-vs-masked A/B strictly paired.
    """
    root = pathlib.Path(root)
    out = {}
    for r in csv.DictReader(open(root / "metadata.csv")):
        parts = r["subject_id"].split("_")          # eda_india_1
        if len(parts) < 3:
            continue
        out[f"{parts[1].capitalize()}_{parts[2]}"] = root / r["filename"]
    return out


class MaskedCropDataset(Dataset):
    """MASKED conjunctiva cut-outs (datasets/eda-masked-crops/): skin removed via
    the alpha channel.

    The transparent background is flattened onto NEUTRAL GREY rather than black:
    a black background creates a hard high-contrast edge that a CNN can latch onto
    as a shape cue, which would be a new shortcut. Grey keeps the boundary soft and
    keeps the mean pixel near the normalisation midpoint.
    """
    def __init__(self, rows, transform, root=MASKED_ROOT, bg=128):
        self.rows = rows
        self.tf = transform
        self.paths = masked_path_map(root)
        self.bg = bg
        missing = [r["subject"] for r in rows if r["subject"] not in self.paths]
        if missing:
            raise KeyError(f"{len(missing)} subjects have no masked crop: {missing[:5]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.paths[r["subject"]]).convert("RGBA")
        bg = Image.new("RGBA", img.size, (self.bg, self.bg, self.bg, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        return self.tf(img), r["y"], r["subject"]


class MaskedTightCropDataset(MaskedCropDataset):
    """Masked conjunctiva, but CROPPED TO THE MASK first so the tissue fills the frame.

    The masked crops ship square-padded with transparency: only ~11% of each image is
    real tissue, the other ~89% is empty padding. Feeding that straight in starves the
    model of resolution on the very thing it must read (~9x fewer tissue pixels than
    the rectangle arm). Cropping to the alpha bounding box first makes the masked arm
    comparable to the rectangle arm in pixels-on-tissue, so the A/B isolates
    'skin removed' instead of confounding it with 'much lower effective resolution'.
    """
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.paths[r["subject"]]).convert("RGBA")
        alpha = np.asarray(img)[..., 3]
        ys, xs = np.where(alpha > 0)
        if xs.size and ys.size:
            img = img.crop((int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1))
        bg = Image.new("RGBA", img.size, (self.bg, self.bg, self.bg, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        return self.tf(img), r["y"], r["subject"]


# selectable input arms for the rectangle-vs-masked A/B
CROPSETS = {
    "rectangle": CropDataset,              # bbox crop, skin included
    "masked": MaskedCropDataset,           # masked, as-shipped (89% padding)
    "masked_tight": MaskedTightCropDataset,  # masked, cropped to tissue (fair comparison)
}


# ---------------------------------------------------------------- colour space
class ToLAB:
    """RGB -> CIELAB, kept as a 3-channel uint8 image.

    Splits lightness (L) from colour (a*, b*), so exposure differences land mostly
    in one channel instead of smearing across all three. a* is the red-green axis --
    the pallor axis anemia actually lives on.
    NOTE: ImageNet normalisation stats no longer strictly apply after this; it's a
    cheap directional test, not a calibrated pipeline.
    """
    def __call__(self, img):
        import cv2
        arr = np.asarray(img.convert("RGB"))
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_RGB2LAB))


class ClaheL:
    """CLAHE applied to the L (lightness) channel ONLY.

    Normalises shadows/exposure while leaving a*/b* untouched, so the colour signal
    survives. This is the cautious version: our gray-world experiment showed that
    normalising the COLOUR channels destroys the pallor cue (-0.10 F1).
    """
    def __init__(self, clip=2.0, grid=8):
        self.clip, self.grid = clip, grid

    def __call__(self, img):
        import cv2
        arr = np.asarray(img.convert("RGB"))
        l, a, b = cv2.split(cv2.cvtColor(arr, cv2.COLOR_RGB2LAB))
        l = cv2.createCLAHE(clipLimit=self.clip,
                            tileGridSize=(self.grid, self.grid)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)


PREPROCS = {"none": None, "lab": ToLAB, "clahe": ClaheL}


def with_preproc(tfs, preproc="none"):
    """Prepend a colour-space step to a (train_tf, eval_tf) pair. Applied to BOTH
    arms because it is preprocessing, not augmentation."""
    if preproc == "none":
        return tfs
    op = PREPROCS[preproc]()
    train_tf, eval_tf = tfs
    return (transforms.Compose([op] + list(train_tf.transforms)),
            transforms.Compose([op] + list(eval_tf.transforms)))


def class_weights(rows):
    """Inverse-frequency weights for WeightedRandomSampler over the given rows."""
    ys = [r["y"] for r in rows]
    n0, n1 = ys.count(0), ys.count(1)
    w = {0: 1.0 / max(n0, 1), 1: 1.0 / max(n1, 1)}
    return torch.tensor([w[y] for y in ys], dtype=torch.double)
