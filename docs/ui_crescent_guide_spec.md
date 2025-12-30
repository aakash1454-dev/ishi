# UI Crescent Guide Specification

## Overview
A camera overlay that guides users to position their lower eyelid conjunctiva within a crescent-shaped region.

## Design

### Visual Guide
```
        ╭─────────────────────╮
       ╱                       ╲
      │     POSITION HERE       │
       ╲_______________________╱
```

- **Shape**: Crescent/arc matching typical conjunctiva shape
- **Color**: Semi-transparent white or green outline (60% opacity)
- **Size**: Adjustable based on device/distance

### User Flow
1. User opens camera
2. Sees crescent overlay with instruction: "Pull down lower eyelid and align"
3. User positions eye so conjunctiva fills the crescent
4. App shows "Ready" indicator when alignment looks good (optional)
5. User taps capture
6. Image is cropped to the crescent region

### Cropping Logic
Once captured, crop the image to the bounding box of the crescent guide:

```python
def crop_to_guide(image, guide_bounds):
    """
    Crop image to the crescent guide region.
    
    guide_bounds: dict with keys 'x', 'y', 'width', 'height'
                  representing the bounding box of the crescent
    """
    x, y, w, h = guide_bounds['x'], guide_bounds['y'], guide_bounds['width'], guide_bounds['height']
    
    # Add small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    
    cropped = image[y:y+h+2*pad, x:x+w+2*pad]
    return cropped
```

## Implementation Notes

### Mobile (React Native / Flutter)
- Use camera overlay component
- SVG or Canvas for crescent shape
- Capture at guide coordinates

### Web
- Use getUserMedia for camera
- Canvas overlay for guide
- Crop using canvas coordinates

## Success Criteria
- User can align in < 5 seconds
- Cropped region contains >80% conjunctiva
- Works in various lighting conditions

## Future Enhancement: Auto-Detection
Once we have enough manually-cropped images, we can:
1. Use them as training data for a segmentation model
2. Add auto-alignment feedback ("move left", "move up")
3. Eventually make the guide optional

