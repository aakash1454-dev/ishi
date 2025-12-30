# Runtime-only model factory. No training imports.
import torch.nn as nn
from typing import Literal, Tuple

Arch = Literal["simple_cnn", "resnet18"]

class SimpleCNN(nn.Module):
    """
    SimpleCNN that matches the trained checkpoint (anemia_cnn.pth).
    Uses self.cnn with layer names: cnn.0, cnn.3, cnn.7, cnn.9
    Input size: 128x128 (from original training)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Must match checkpoint layer names: cnn.0, cnn.3, etc.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # cnn.0, cnn.1, cnn.2
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # cnn.3, cnn.4, cnn.5
            nn.Flatten(),                                                 # cnn.6
            nn.Linear(32 * 32 * 32, 64),  # 128/4=32, so 32*32*32         # cnn.7
            nn.ReLU(),                                                    # cnn.8
            nn.Linear(64, num_classes)                                    # cnn.9
        )

    def forward(self, x):
        return self.cnn(x)

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
