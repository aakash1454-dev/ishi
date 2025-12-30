#!/usr/bin/env python3
"""
Simplified anemia prediction using ResNet18 only.
Drops MobileNet (which was broken) and uses optimized threshold.

Usage:
    python scripts/predict_resnet_only.py --image path/to/image.png
    python scripts/predict_resnet_only.py --csv path/to/images.csv --out results.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# =============================================================================
# CONFIGURATION - Easy to adjust
# =============================================================================
DEFAULT_THRESHOLD = 0.20  # Optimized for ~80% sensitivity
DEFAULT_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_resnet18(checkpoint_path: str, num_classes: int = 2):
    """Load ResNet18 model from checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def get_transform(img_size: int = DEFAULT_IMG_SIZE):
    """Get inference transforms (must match training!)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path: str) -> Image.Image:
    """Load image and convert RGBA to RGB with black background."""
    img = Image.open(path)
    
    if img.mode == 'RGBA':
        # Convert transparent background to black
        background = Image.new('RGB', img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background
    
    return img.convert('RGB')


# =============================================================================
# PREDICTION
# =============================================================================
def predict_single(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu"
) -> dict:
    """Predict anemia for a single image."""
    
    # Load and preprocess
    img = load_image(image_path)
    x = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(x)
        
        # Handle both 1-output (sigmoid) and 2-output (softmax) models
        if logits.shape[-1] == 1:
            prob = torch.sigmoid(logits).item()
        else:
            probs = F.softmax(logits, dim=-1)
            prob = probs[0, 1].item()  # Probability of class 1 (anemic)
    
    # Apply threshold
    is_anemic = prob >= threshold
    
    return {
        "filepath": image_path,
        "prob_anemic": round(prob, 4),
        "prediction": "anemic" if is_anemic else "not_anemic",
        "threshold_used": threshold,
    }


def predict_batch(
    model: torch.nn.Module,
    image_paths: list,
    transform: transforms.Compose,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu"
) -> list:
    """Predict anemia for multiple images."""
    results = []
    
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
            
        try:
            result = predict_single(model, path, transform, threshold, device)
            results.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"[ERR] {path}: {e}")
            
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ResNet-only anemia prediction")
    parser.add_argument("--model", required=True, help="Path to ResNet18 checkpoint")
    parser.add_argument("--image", help="Single image to predict")
    parser.add_argument("--csv", help="CSV file with 'filepath' column")
    parser.add_argument("--out", help="Output CSV path (for batch mode)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    args = parser.parse_args()
    
    # Validate args
    if not args.image and not args.csv:
        parser.error("Must provide --image or --csv")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_resnet18(args.model)
    model.to(device)
    
    transform = get_transform(args.img_size)
    
    # Single image mode
    if args.image:
        result = predict_single(model, args.image, transform, args.threshold, device)
        print("\nResult:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return
    
    # Batch mode
    if args.csv:
        # Load image paths from CSV
        image_paths = []
        with open(args.csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get('filepath') or row.get('path') or row.get('image')
                if path:
                    image_paths.append(path)
        
        print(f"Processing {len(image_paths)} images...")
        results = predict_batch(model, image_paths, transform, args.threshold, device)
        
        # Save results
        out_path = args.out or "predictions_resnet_only.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "prob_anemic", "prediction", "threshold_used"])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nSaved {len(results)} predictions to {out_path}")
        
        # Summary
        n_anemic = sum(1 for r in results if r["prediction"] == "anemic")
        print(f"Predicted anemic: {n_anemic}/{len(results)} ({100*n_anemic/len(results):.1f}%)")


if __name__ == "__main__":
    main()

