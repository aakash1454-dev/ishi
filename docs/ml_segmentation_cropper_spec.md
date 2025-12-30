# ML Segmentation Cropper Specification

## Overview
A U-Net style segmentation model that automatically detects and crops the conjunctiva region.

## Architecture

```
Input Image (224x224x3)
        │
        ▼
┌───────────────────┐
│   U-Net Encoder   │  ← ResNet18 backbone (pretrained)
│   (downsample)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   U-Net Decoder   │  ← Upsampling + skip connections
│   (upsample)      │
└───────────────────┘
        │
        ▼
Output Mask (224x224x1)  ← Binary: 1=conjunctiva, 0=background
```

## Training Data Required

To train this model, you need:
- **~200-500 images** with manually annotated conjunctiva masks
- **Annotation tool**: LabelMe, CVAT, or Roboflow

### Annotation Process
1. Load eye image
2. Draw polygon around conjunctiva region
3. Export as binary mask (PNG)

### Data Format
```
data/
  conjunctiva_segmentation/
    images/
      001.png
      002.png
      ...
    masks/
      001.png  # Binary mask: white=conjunctiva, black=background
      002.png
      ...
```

## Model Code (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models

class ConjunctivaSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use pretrained ResNet18 as encoder
        resnet = models.resnet18(pretrained=True)
        
        # Encoder (remove final FC layers)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 64
        self.enc3 = resnet.layer2  # 128
        self.enc4 = resnet.layer3  # 256
        self.enc5 = resnet.layer4  # 512
        
        # Decoder
        self.dec4 = self._decoder_block(512, 256)
        self.dec3 = self._decoder_block(256 + 256, 128)
        self.dec2 = self._decoder_block(128 + 128, 64)
        self.dec1 = self._decoder_block(64 + 64, 64)
        
        # Final output
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(e5)
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        
        # Upsample to original size and predict
        d1 = nn.functional.interpolate(d1, scale_factor=4, mode='bilinear')
        out = torch.sigmoid(self.final(d1))
        
        return out
```

## Training Script

```python
# train_segmenter.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ConjunctivaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask.resize((224, 224)))
        
        return image, mask

def train():
    model = ConjunctivaSegmenter()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ConjunctivaDataset('data/images', 'data/masks', transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for epoch in range(50):
        for images, masks in loader:
            pred = model(images)
            loss = criterion(pred, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'conjunctiva_segmenter.pth')
```

## Inference

```python
def segment_and_crop(model, image_path):
    """Segment conjunctiva and crop to bounding box."""
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess
    x = transform(img).unsqueeze(0)
    
    # Predict mask
    with torch.no_grad():
        mask = model(x)[0, 0].numpy()
    
    # Threshold
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Find bounding box
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None  # No conjunctiva found
    
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    
    # Crop original image
    img_np = np.array(img.resize((224, 224)))
    cropped = img_np[y1:y2, x1:x2]
    
    return cropped
```

## Time Estimate
- **Data annotation**: 2-4 hours for 200 images
- **Training**: 1-2 hours on GPU
- **Integration**: 1 day
- **Total**: ~1 week

## Alternative: Use Pretrained Model
Instead of training from scratch, consider:
1. **SAM (Segment Anything)** - Meta's model, works zero-shot
2. **MediaPipe Face Mesh** - Can detect eye landmarks
3. **Roboflow** - Train custom model with their platform

