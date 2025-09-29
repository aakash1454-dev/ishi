# Runtime-only model factory. No training imports.
import torch.nn as nn
from typing import Literal, Tuple

Arch = Literal["simple_cnn", "resnet18"]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def build_model(arch: Arch, num_classes: int = 2) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif arch == "resnet18":
        # import inside to avoid pulling all torchvision at import time
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=None)  # runtime loads your checkpoint, not ImageNet
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported arch: {arch}")
