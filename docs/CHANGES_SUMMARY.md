# ISHI Anemia Detection - Technical Changes Summary

**Date:** December 28, 2025  
**Author:** Abil Hamded

---

## Summary

The anemia detection system was producing incorrect results, classifying healthy individuals (with red/pink conjunctiva) as anemic. This document details all changes made to resolve these issues across the mobile application and backend API.

---

## Part 1: Mobile Application Changes

### 1.1 Android Build Configuration

**Files Modified:**
- `mobile/android/app/build.gradle.kts`
- `mobile/android/settings.gradle.kts`

**Changes:**
- Updated `compileSdk` from version 35 to version 36
- Updated Android Gradle Plugin from version 8.7.3 to version 8.9.1
- These updates were required for compatibility with newer Flutter plugins

### 1.2 Camera FileProvider Configuration

**Files Modified:**
- `mobile/android/app/src/main/AndroidManifest.xml`
- `mobile/android/app/src/main/res/xml/file_paths.xml` (new file)

**Changes:**
- Added FileProvider declaration in AndroidManifest.xml for camera image capture functionality
- Created file_paths.xml to define accessible storage paths
- Set authority to `${applicationId}.flutter.image_provider` to match image_picker plugin requirements

### 1.3 User Interface Crescent Guide

**File Created:**
- `mobile/lib/widgets/conjunctiva_guide.dart`

**Features Added:**
- Crescent-shaped overlay widget to guide users in positioning their lower eyelid
- Instruction card component with step-by-step capture instructions
- Visual alignment markers for improved image capture accuracy

### 1.4 Adjustable Image Viewer

**File Modified:**
- `mobile/lib/pages/camera_page.dart`

**Features Added:**
- Pinch-to-zoom functionality on captured images (0.5x to 4x scale)
- Draggable crescent guide overlay that users can position over the conjunctiva
- Resizable guide with increment/decrement buttons (30% to 200% scale)
- Reset button to restore default guide position and size
- 300px image viewing area with rounded borders

---

## Part 2: Backend API Changes

### 2.1 SimpleCNN Architecture Correction

**File Modified:**
- `api/models/factory.py`

**Problem:**
The SimpleCNN class definition used layer names (`self.features`) that did not match the trained checkpoint file (`self.cnn`), causing weights to fail loading.

**Before:**
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)
```

**After:**
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
```

### 2.2 Image Preprocessing Correction

**File Modified:**
- `api/routes/anemia.py`

**Problem:**
The API was applying ImageNet normalization to input images, but the SimpleCNN model was trained without any normalization.

**Before:**
```python
def _make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

**After:**
```python
def _make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
```

### 2.3 Class Index Correction

**File Modified:**
- `api/routes/anemia.py`

**Problem:**
The prediction function was using the wrong class index and applying an unnecessary probability inversion based on analysis of a different model (ResNet).

**Before:**
```python
def _predict_p_anemic(pil_img: Image.Image) -> float:
    # ... model inference ...
    idx_anemic = _classes.index("anemic")  # Returns 0
    raw_prob = float(probs[idx_anemic])
    return 1.0 - raw_prob  # Incorrect inversion
```

**After:**
```python
def _predict_p_anemic(pil_img: Image.Image) -> float:
    # ... model inference ...
    # SimpleCNN: class 0 = not anemic, class 1 = anemic
    return float(probs[1])
```

### 2.4 Threshold Configuration

**File Modified:**
- `api/utils/config.py`

**Change:**
Updated default threshold from 0.303 to 0.60 (60%) to reduce false positive rate.

---

## Part 3: Runtime Configuration

### Required Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| ANEMIA_CKPT | models/anemia/anemia_cnn.pth | Path to model checkpoint |
| ANEMIA_THRESHOLD | 0.60 | Classification threshold |
| ANEMIA_IMG_SIZE | 128 | Input image size (must match training) |

### Backend Startup Command

```powershell
cd C:\Desktop\ironstrong-health\ishi
$env:ANEMIA_CKPT="models/anemia/anemia_cnn.pth"
$env:ANEMIA_THRESHOLD="0.60"
$env:ANEMIA_IMG_SIZE="128"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Mobile Application Startup Command

```powershell
cd C:\Desktop\ironstrong-health\ishi\mobile
flutter run -d <device_id> --dart-define=API_BASE_URL=http://<PC_IP_ADDRESS>:8000
```

---

## Part 4: Root Cause Analysis

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| Model weights not loading | Layer name mismatch between code and checkpoint | Changed `self.features` to `self.cnn` |
| Incorrect predictions | ImageNet normalization applied to model trained without it | Removed normalization transform |
| Class inversion | Using probs[0] instead of probs[1] for anemic class | Fixed to use probs[1] |
| Matrix multiplication error | Image size 224x224 vs model trained on 128x128 | Set ANEMIA_IMG_SIZE=128 |
| High false positive rate | Threshold too low at 0.303 | Increased to 0.60 |

---

## Part 5: Current System State

**Model:** SimpleCNN (anemia_cnn.pth)  
**Input Size:** 128 x 128 pixels  
**Preprocessing:** Resize and ToTensor only (no normalization)  
**Threshold:** 60%  
**Classification Logic:** Score >= 60% indicates Anemic, Score < 60% indicates Not Anemic

---

## Part 6: Recommendations for Future Work

1. **Model Retraining:** Consider retraining with a more robust architecture (ResNet18) on a larger, balanced dataset.

2. **Cropper Integration:** The pseudomask cropper script path needs to be updated for Windows environments to enable automatic conjunctiva detection.

3. **Threshold Calibration:** Perform systematic threshold calibration on a held-out validation set to optimize sensitivity and specificity.

4. **Model Checkpointing:** Ensure future trained models are saved with architecture metadata to prevent layer name mismatches.

---

## Simplified Summary

The app was saying everyone had anemia because the code was looking at the wrong number from the AI model and using settings that did not match how the model was trained.The code was fixed to read the correct output, removed extra image processing that confused the model, and raised the threshold so only scores above 60% are flagged as anemic. The mobile app was also updated to help users take better pictures of their eye.
