# Quick Reference: `predict_image` Method

## Summary
Apply trained color correction models to new images without requiring a color checker chart.

## Signature
```python
def predict_image(
    self, 
    Image: Union[str, np.ndarray], 
    show: bool = False
) -> Dict[str, np.ndarray]
```

## Inputs

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `Image` | `str` or `np.ndarray` | Yes | - | Input image as file path or RGB array (float64, 0-1 range) |
| `show` | `bool` | No | `False` | Display plots of each correction stage |

## Outputs

Returns a dictionary with the following structure:

```python
{
    'FFC': np.ndarray,  # Flat-field corrected image (always present)
    'GC': np.ndarray,   # Gamma corrected image (always present)
    'WB': np.ndarray,   # White-balanced image (always present)
    'CC': np.ndarray or None  # Final color corrected image (None if no CC model)
}
```

All output arrays have:
- **Shape**: `(height, width, 3)`
- **dtype**: `float64`
- **Range**: `[0, 1]`
- **Color space**: RGB

## Quick Start Example

```python
from ColorCorrectionPipeline.ccp import ColorCorrection
from ColorCorrectionPipeline.Configs.configs import Config

# 1. Train the pipeline first
cc = ColorCorrection()
config = Config(do_ffc=True, do_gc=True, do_wb=True, do_cc=True)
cc.run(Image="training_image.jpg", White_Image="white.jpg", config=config)

# 2. Apply to new images
results = cc.predict_image(Image="test_image.jpg")

# 3. Access results
final_image = results['CC']  # Get final corrected image
```

## Pipeline Flow

```
┌─────────────────────┐
│   Input Image       │ ← File path (str) or array (np.ndarray)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Flat-Field         │ → results['FFC']
│  Correction (FFC)   │   (corrects uneven illumination)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Gamma              │ → results['GC']
│  Correction (GC)    │   (corrects brightness/contrast)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  White              │ → results['WB']
│  Balance (WB)       │   (corrects color temperature)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Color              │ → results['CC']
│  Correction (CC)    │   (final color mapping)
└─────────────────────┘
```

## Input Format Details

### Option 1: File Path (String)
```python
# Any OpenCV-supported format
results = cc.predict_image(Image="image.jpg")
results = cc.predict_image(Image="image.png")
results = cc.predict_image(Image="/path/to/image.tiff")
```

### Option 2: NumPy Array
```python
import cv2
from ColorCorrectionPipeline.key_functions import to_float64

# Load and convert to correct format
img_bgr = cv2.imread("image.jpg")
img_rgb = to_float64(img_bgr[:, :, ::-1])  # BGR→RGB, scale to [0,1]

results = cc.predict_image(Image=img_rgb)
```

## Output Usage Examples

### Save Final Result
```python
import cv2

results = cc.predict_image(Image="test.jpg")

if results['CC'] is not None:
    # Convert from RGB float [0,1] to BGR uint8 [0,255]
    output_bgr = (results['CC'][:, :, ::-1] * 255).astype('uint8')
    cv2.imwrite("corrected.jpg", output_bgr)
```

### Compare Stages
```python
import matplotlib.pyplot as plt

results = cc.predict_image(Image="test.jpg")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(results['FFC'])
axes[0, 0].set_title('Flat-Field Corrected')
axes[0, 1].imshow(results['GC'])
axes[0, 1].set_title('Gamma Corrected')
axes[1, 0].imshow(results['WB'])
axes[1, 0].set_title('White Balanced')
if results['CC'] is not None:
    axes[1, 1].imshow(results['CC'])
    axes[1, 1].set_title('Color Corrected')
plt.tight_layout()
plt.show()
```

### Process Batch
```python
import glob

image_files = glob.glob("images/*.jpg")

for img_path in image_files:
    results = cc.predict_image(Image=img_path)
    
    if results['CC'] is not None:
        output_name = f"corrected_{os.path.basename(img_path)}"
        output_bgr = (results['CC'][:, :, ::-1] * 255).astype('uint8')
        cv2.imwrite(output_name, output_bgr)
```

## Common Issues

### ❌ Error: "Cannot read Image from..."
**Cause**: Invalid file path
```python
# Solution: Check path exists
import os
if os.path.exists(image_path):
    results = cc.predict_image(Image=image_path)
```

### ❌ Error: "Image must be a file path or numpy array"
**Cause**: Wrong input type
```python
# Solution: Use string path or numpy array
results = cc.predict_image(Image="image.jpg")  # ✓ Correct
results = cc.predict_image(Image=img_array)    # ✓ Correct
results = cc.predict_image(Image=123)          # ✗ Wrong
```

### ⚠️ Warning: Results contain None values
**Cause**: Models not trained
```python
# Solution: Train models first
cc.run(Image=training_img, config=Config(do_cc=True))
# Then predict
results = cc.predict_image(Image=test_img)
```

## Performance Tips

1. **Use NumPy arrays for batch processing** (avoid repeated file I/O)
2. **GPU acceleration** is automatic for gamma correction if CUDA available
3. **Memory usage**: ~5× image size (one copy per stage)
4. **Timing**: Logged automatically in console

## Prerequisites Checklist

Before using `predict_image`:
- [ ] Created `ColorCorrection` instance
- [ ] Trained models using `run()` OR loaded saved models
- [ ] (Optional) Set reference illuminant via `get_reference_values()`

## See Full Documentation
For detailed information, see [predict_image_analysis.md](./predict_image_analysis.md)
