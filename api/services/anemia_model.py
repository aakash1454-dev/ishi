# api/services/anemia_model.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

# Constants
MODEL_PATH = os.path.join("api", "models", "anemia_cnn.pth")
IMAGE_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AnemiaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE[0] // 4) * (IMAGE_SIZE[1] // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.cnn(x)

# Load model
model = AnemiaNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def predict_anemia(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return {"anemic": bool(pred)}
