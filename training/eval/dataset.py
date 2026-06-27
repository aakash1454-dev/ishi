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


class CropDataset(Dataset):
    """rows: list of folds.csv dicts. Returns (image_tensor, y, subject)."""
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


def class_weights(rows):
    """Inverse-frequency weights for WeightedRandomSampler over the given rows."""
    ys = [r["y"] for r in rows]
    n0, n1 = ys.count(0), ys.count(1)
    w = {0: 1.0 / max(n0, 1), 1: 1.0 / max(n1, 1)}
    return torch.tensor([w[y] for y in ys], dtype=torch.double)
