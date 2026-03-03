#!/usr/bin/env python3
"""
Optimized Anemia Detection Training Script v2
- Designed for small datasets (~200-500 images)
- Strong augmentation pipeline for medical imaging
- K-fold cross-validation for reliable metrics
- Learning rate scheduling with warmup
- Early stopping to prevent overfitting
- Test-time augmentation for better predictions
"""

import argparse
import os
import random
from pathlib import Path
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, 
    accuracy_score, roc_auc_score, classification_report
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# ========================
# Configuration
# ========================

class Config:
    """Training configuration - tune these for your dataset"""
    
    # Data
    IMG_SIZE = 224
    VAL_RATIO = 0.2
    
    # Training
    EPOCHS = 50              # More epochs, but early stopping will kick in
    BATCH_SIZE = 16          # Smaller batch = more gradient updates
    LR = 1e-4                # Lower LR for fine-tuning pretrained
    WEIGHT_DECAY = 1e-4
    
    # Early stopping
    PATIENCE = 10            # Stop if no improvement for N epochs
    MIN_DELTA = 0.001        # Minimum improvement to count
    
    # Augmentation strength (0.0 = off, 1.0 = maximum)
    AUG_STRENGTH = 0.8       # Strong augmentation for small dataset
    
    # Label smoothing (reduces overconfidence)
    LABEL_SMOOTHING = 0.1
    
    # Seed
    SEED = 42


# ========================
# Utilities
# ========================

def set_seed(seed: int):
    """Reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


# ========================
# Data Augmentation
# ========================

def build_transforms(img_size: int, aug_strength: float = 0.8, is_train: bool = True):
    """
    Build augmentation pipeline optimized for conjunctiva images.
    
    Key augmentations for anemia detection:
    - Color jitter: Critical! Must learn to distinguish pale vs red
    - Rotation: Eyes can be photographed at various angles
    - Affine: Slight perspective changes from different phone angles
    - Blur: Robustness to focus variations
    """
    
    if is_train and aug_strength > 0:
        # Strong augmentation for training
        aug = aug_strength
        
        transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for crop
            transforms.RandomResizedCrop(
                img_size, 
                scale=(0.8, 1.0),      # Random zoom 80-100%
                ratio=(0.9, 1.1)       # Slight aspect ratio variation
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Eyes can be upside down
            transforms.RandomRotation(
                degrees=int(20 * aug),  # Up to ±20°
                fill=0
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1 * aug, 0.1 * aug),  # Slight translation
                scale=(1.0 - 0.1 * aug, 1.0 + 0.1 * aug),
                fill=0
            ),
            # COLOR AUGMENTATION - Critical for anemia detection!
            transforms.ColorJitter(
                brightness=0.3 * aug,   # Lighting variations
                contrast=0.3 * aug,     # Camera differences
                saturation=0.4 * aug,   # KEY: pale vs vibrant red
                hue=0.1 * aug           # Slight color shifts
            ),
            transforms.RandomGrayscale(p=0.05),  # Occasional grayscale
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Focus variation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(
                p=0.2 * aug,            # Randomly erase patches
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3)
            ),
        ])
    else:
        # Clean transform for validation/inference
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform


def get_tta_transforms(img_size: int):
    """
    Test-Time Augmentation transforms.
    Returns list of transforms to average predictions over.
    """
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return [
        # Original
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Slight rotation +10
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Slight rotation -10
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            base_norm,
        ]),
    ]


# ========================
# Model Building
# ========================

def build_model(backbone: str = "resnet18", num_classes: int = 2, dropout: float = 0.3):
    """
    Build model with proper initialization for fine-tuning.
    
    Options:
    - resnet18: Good balance of speed and accuracy
    - resnet50: More capacity, might overfit on small data
    - efficientnet_b0: Modern, efficient architecture
    - mobilenet_v3: Fast inference for mobile
    """
    
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze early layers (they learn general features)
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        # Replace classifier with dropout + new FC
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in list(model.parameters())[:-30]:
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in list(model.parameters())[:-15]:
            param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return model


# ========================
# Training Components
# ========================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_model = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            
        return self.should_stop


def make_weighted_sampler(dataset, indices=None):
    """Create weighted sampler for class imbalance"""
    if indices is None:
        indices = list(range(len(dataset)))
    
    # Get labels
    labels = []
    for i in indices:
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels.append(label)
    
    # Compute weights
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[l] for l in labels]
    
    return WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )


def train_one_epoch(model, loader, optimizer, scheduler, device, label_smoothing=0.0):
    """Train for one epoch with label smoothing"""
    model.train()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    
    # Step scheduler after epoch
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """Evaluate model and return metrics"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1
            total_loss += loss.item() * x.size(0)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }


def evaluate_with_tta(model, dataset, indices, device, img_size):
    """Evaluate with Test-Time Augmentation"""
    model.eval()
    
    tta_transforms = get_tta_transforms(img_size)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for idx in indices:
            # Get original image (before transforms)
            img_path, label = dataset.samples[idx]
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Average predictions over TTA transforms
            prob_sum = 0.0
            for tf in tta_transforms:
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                prob_sum += probs[0, 1].item()  # Prob of anemic
            
            avg_prob = prob_sum / len(tta_transforms)
            all_labels.append(label)
            all_probs.append(avg_prob)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
    }


# ========================
# Visualization
# ========================

def plot_training_history(history, out_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], label='Val AUC', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC-ROC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ========================
# Main Training
# ========================

def train_single_fold(
    train_dataset, val_dataset, 
    train_indices, val_indices,
    config, device, out_dir, fold=None
):
    """Train a single fold"""
    
    fold_str = f"Fold {fold}" if fold else "Training"
    print(f"\n{'='*50}")
    print(f"{fold_str}")
    print(f"{'='*50}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Create data subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Weighted sampler for training
    train_labels = [train_dataset.samples[i][1] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=2)
    print(f"Class distribution - Not Anemic: {class_counts[0]}, Anemic: {class_counts[1]}")
    
    sampler = make_weighted_sampler(train_dataset, train_indices)
    
    # Data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    model = build_model(backbone=config.BACKBONE, num_classes=2, dropout=0.3)
    model = model.to(device)
    
    # Optimizer with layer-wise learning rates
    # Higher LR for classifier, lower for backbone
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LR * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': config.LR}       # Full LR for classifier
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    total_steps = config.EPOCHS * len(train_loader)
    warmup_steps = len(train_loader) * 3  # 3 epochs warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))  # Cosine decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, None, device,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # Step scheduler
        scheduler.step()
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print progress
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Track best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_cm = val_metrics['confusion_matrix']
            
            # Save best model
            fold_suffix = f"_fold{fold}" if fold else ""
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'f1': best_f1,
                'accuracy': val_metrics['accuracy'],
                'auc': val_metrics['auc'],
                'classes': ['nonanemic', 'anemic'],  # class 0, class 1
                'backbone': config.BACKBONE,
                'img_size': config.IMG_SIZE,
            }, out_dir / f"best{fold_suffix}.pth")
        
        # Early stopping check
        if early_stopping(val_metrics['f1'], model):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print(f"\nBest F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    model.load_state_dict(early_stopping.best_model)
    
    # Save plots
    fold_suffix = f"_fold{fold}" if fold else ""
    plot_training_history(history, out_dir / f"training_curves{fold_suffix}.png")
    plot_confusion_matrix(best_cm, ['Not Anemic', 'Anemic'], out_dir / f"confusion_matrix{fold_suffix}.png")
    
    # TTA evaluation
    print("\nEvaluating with Test-Time Augmentation...")
    tta_metrics = evaluate_with_tta(model, val_dataset, val_indices, device, config.IMG_SIZE)
    print(f"TTA Results - Acc: {tta_metrics['accuracy']:.4f}, "
          f"F1: {tta_metrics['f1']:.4f}, AUC: {tta_metrics['auc']:.4f}")
    
    return {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'history': history,
        'tta_metrics': tta_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Optimized Anemia Detection Training")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data folder with anemic/ and nonanemic/ subfolders")
    parser.add_argument("--out", type=str, default="runs/anemia_v2",
                       help="Output directory")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kfold", type=int, default=0,
                       help="Number of K-fold CV splits (0 = no CV, single split)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup config
    config = Config()
    config.BACKBONE = args.backbone
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.SEED = args.seed
    
    set_seed(config.SEED)
    
    # Output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check data directory
    data_path = Path(args.data)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Load dataset
    print(f"\nLoading data from: {data_path}")
    
    train_tf = build_transforms(config.IMG_SIZE, aug_strength=config.AUG_STRENGTH, is_train=True)
    val_tf = build_transforms(config.IMG_SIZE, is_train=False)
    
    # Create datasets with different transforms
    train_dataset = datasets.ImageFolder(str(data_path), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(data_path), transform=val_tf)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to index: {train_dataset.class_to_idx}")
    print(f"Total images: {len(train_dataset)}")
    
    # Class distribution
    labels = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(labels)
    for cls_name, count in zip(train_dataset.classes, class_counts):
        print(f"  {cls_name}: {count}")
    
    # K-fold or single split
    if args.kfold > 1:
        print(f"\n{'='*50}")
        print(f"Running {args.kfold}-Fold Cross Validation")
        print(f"{'='*50}")
        
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=config.SEED)
        
        fold_results = []
        all_indices = list(range(len(train_dataset)))
        labels = [train_dataset.samples[i][1] for i in all_indices]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, labels), 1):
            result = train_single_fold(
                train_dataset, val_dataset,
                train_idx.tolist(), val_idx.tolist(),
                config, device, out_dir, fold=fold
            )
            fold_results.append(result)
        
        # Summary
        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        
        f1_scores = [r['best_f1'] for r in fold_results]
        tta_f1_scores = [r['tta_metrics']['f1'] for r in fold_results]
        
        print(f"F1 Scores: {[f'{f:.4f}' for f in f1_scores]}")
        print(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"\nTTA F1 Scores: {[f'{f:.4f}' for f in tta_f1_scores]}")
        print(f"Mean TTA F1: {np.mean(tta_f1_scores):.4f} ± {np.std(tta_f1_scores):.4f}")
        
    else:
        # Single train/val split
        print(f"\nUsing {config.VAL_RATIO*100:.0f}% validation split")
        
        all_indices = list(range(len(train_dataset)))
        labels = [train_dataset.samples[i][1] for i in all_indices]
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            all_indices, 
            test_size=config.VAL_RATIO,
            stratify=labels,
            random_state=config.SEED
        )
        
        result = train_single_fold(
            train_dataset, val_dataset,
            train_idx, val_idx,
            config, device, out_dir
        )
    
    print(f"\n✅ Training complete! Results saved to: {out_dir}")
    print("\nTo use this model in the API, copy best.pth to api/models/")


if __name__ == "__main__":
    main()
