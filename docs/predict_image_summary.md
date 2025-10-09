# Summary: `predict_image` Method Analysis

## Executive Summary

The `predict_image` method is the inference/prediction function of the ColorCorrectionPipeline package. After training correction models using the `run()` method, `predict_image` applies those models to new images without requiring a color checker chart.

---

## Inputs

### Parameter 1: `Image` (Required)
**Type:** `Union[str, np.ndarray]`

**Accepted Values:**
- **File path (string)**: `"path/to/image.jpg"`, `"/absolute/path/image.png"`, etc.
  - Any format supported by OpenCV (JPEG, PNG, TIFF, BMP, etc.)
  - Automatically loaded and converted to RGB float64 [0,1]
- **NumPy array**: Pre-loaded image
  - Shape: `(height, width, 3)`
  - dtype: `float64`
  - Range: `[0, 1]`
  - Color order: RGB (not BGR)

**Validation:**
- Raises `FileNotFoundError` if file path cannot be read
- Raises `TypeError` if input is neither string nor numpy array

### Parameter 2: `show` (Optional)
**Type:** `bool`  
**Default:** `False`

**Purpose:** Control visualization of correction stages
- `True`: Display plots for each stage using `colour.plotting.plot_image()`
- `False`: No visualization (silent operation)

**Note:** Visualization errors are caught and ignored to prevent pipeline failure

---

## Outputs

### Return Type
**Type:** `Dict[str, np.ndarray]`

### Dictionary Structure

```python
{
    'FFC': np.ndarray,        # Flat-Field Corrected image
    'GC': np.ndarray,         # Gamma Corrected image
    'WB': np.ndarray,         # White Balanced image
    'CC': np.ndarray or None  # Color Corrected image (final result)
}
```

### Output Specifications

All output arrays (when not None) have:
- **Shape:** `(height, width, 3)` - same as input
- **dtype:** `float64`
- **Range:** `[0, 1]` - clipped to valid range
- **Color Space:** RGB

### Output Keys Explained

| Key | Full Name | Description | Availability |
|-----|-----------|-------------|--------------|
| `'FFC'` | Flat-Field Correction | Corrects uneven illumination across the image field | Always present |
| `'GC'` | Gamma Correction | Corrects brightness/contrast using polynomial mapping | Always present |
| `'WB'` | White Balance | Corrects color temperature using diagonal matrix | Always present |
| `'CC'` | Color Correction | Final color mapping using either conventional or ML method | Always present (may be `None`) |

---

## Processing Pipeline

The method applies corrections sequentially, with each stage depending on the previous:

```
Input Image (RGB, float64, [0,1])
    ↓
[1] Flat-Field Correction (FFC)
    ├─ If model_ffc exists: Apply multiplier to correct illumination
    └─ If model_ffc is None: Pass through unchanged
    ↓ results['FFC']
    ↓
[2] Gamma Correction (GC)
    ├─ If model_gc exists: Apply polynomial correction to luminance
    └─ If model_gc is None: Pass through unchanged
    ↓ results['GC']
    ↓
[3] White Balance (WB)
    ├─ If model_wb exists: Apply diagonal matrix multiplication
    └─ If model_wb is None: Pass through unchanged
    ↓ results['WB']
    ↓
[4] Color Correction (CC)
    ├─ If model_cc exists:
    │   ├─ 'conv' method: Apply color correction matrix (Finlayson 2015)
    │   └─ 'ours' method: Apply ML model (linear/PLS/NN)
    └─ If model_cc is None: Set to None
    ↓ results['CC']
    ↓
Output Dictionary
```

---

## Complete Usage Example

```python
import numpy as np
import cv2
from ColorCorrectionPipeline.ccp import ColorCorrection
from ColorCorrectionPipeline.Configs.configs import Config
from ColorCorrectionPipeline.key_functions import to_float64

# ============================================================================
# STEP 1: Initialize and Train Models
# ============================================================================

cc = ColorCorrection()

# Configure pipeline
config = Config(
    do_ffc=True,   # Enable flat-field correction
    do_gc=True,    # Enable gamma correction
    do_wb=True,    # Enable white balance
    do_cc=True,    # Enable color correction
    save=False,
    FFC_kwargs={},
    GC_kwargs={'max_degree': 5},
    WB_kwargs={},
    CC_kwargs={'cc_method': 'ours', 'mtd': 'linear', 'degree': 3}
)

# Train on image with color checker
metrics, images, errors = cc.run(
    Image="training_image_with_colorchecker.jpg",
    White_Image="white_background.jpg",  # Optional for FFC
    name_="training",
    config=config
)

print("Training complete! Models are ready for prediction.")

# ============================================================================
# STEP 2: Apply to New Images - Method 1 (File Path)
# ============================================================================

# Simple usage with file path
results = cc.predict_image(Image="test_image_1.jpg", show=False)

# Access intermediate stages
flat_field_corrected = results['FFC']  # After illumination correction
gamma_corrected = results['GC']        # After brightness/contrast correction
white_balanced = results['WB']         # After color temperature correction
final_corrected = results['CC']        # Final result (or None)

# Save final result
if final_corrected is not None:
    output_bgr = (final_corrected[:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite("corrected_image_1.jpg", output_bgr)

# ============================================================================
# STEP 3: Apply to New Images - Method 2 (NumPy Array)
# ============================================================================

# Load image as numpy array
img_bgr = cv2.imread("test_image_2.jpg")
img_rgb = to_float64(img_bgr[:, :, ::-1])  # Convert BGR→RGB, scale to [0,1]

# Apply corrections
results = cc.predict_image(Image=img_rgb, show=False)

# Extract just the final result
final_image = results['CC']

# ============================================================================
# STEP 4: Batch Processing
# ============================================================================

import glob
import os

# Process all images in a directory
image_paths = glob.glob("test_images/*.jpg")

for img_path in image_paths:
    print(f"Processing {img_path}...")
    
    # Apply corrections
    results = cc.predict_image(Image=img_path, show=False)
    
    # Save final corrected image
    if results['CC'] is not None:
        basename = os.path.basename(img_path)
        output_path = f"corrected_{basename}"
        output_bgr = (results['CC'][:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(output_path, output_bgr)
        print(f"  → Saved {output_path}")

# ============================================================================
# STEP 5: Compare All Stages
# ============================================================================

import matplotlib.pyplot as plt

results = cc.predict_image(Image="test_image.jpg", show=False)

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].imshow(results['FFC'])
axes[0, 0].set_title('1. Flat-Field Corrected')
axes[0, 0].axis('off')

axes[0, 1].imshow(results['GC'])
axes[0, 1].set_title('2. Gamma Corrected')
axes[0, 1].axis('off')

axes[1, 0].imshow(results['WB'])
axes[1, 0].set_title('3. White Balanced')
axes[1, 0].axis('off')

if results['CC'] is not None:
    axes[1, 1].imshow(results['CC'])
    axes[1, 1].set_title('4. Color Corrected (Final)')
else:
    axes[1, 1].text(0.5, 0.5, 'No CC Model', ha='center', va='center')
    axes[1, 1].set_title('4. Color Corrected')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('correction_stages_comparison.png', dpi=150)
plt.show()
```

---

## Key Points

### ✅ What It Does
- Applies pre-trained correction models to new images
- Processes images without requiring color checker chart
- Returns intermediate results for each correction stage
- Maintains image dimensions and RGB color space
- Automatically clips values to valid [0,1] range

### ⚠️ What It Requires
- Models must be trained first using `run()` OR loaded from disk
- Reference illuminant must be set (done automatically during training)
- Input must be valid image (file path or RGB array)

### 📊 Performance
- Logs total prediction time automatically
- Uses GPU acceleration for gamma correction (if CUDA available)
- Memory usage: ~5× input image size (one copy per stage)
- Typical time: < 1 second for 1920×1080 image on GPU

### 🔧 Flexibility
- Can skip stages by not training corresponding models
- Each output stage is independent and usable
- Silent operation by default (no visualization)
- Batch processing friendly

---

## Input/Output Type Summary Table

| Aspect | Details |
|--------|---------|
| **Input Type** | `str` (file path) OR `np.ndarray` (RGB image) |
| **Input Shape** | Any `(H, W, 3)` |
| **Input dtype** | `float64` (if array) |
| **Input Range** | `[0, 1]` (if array) |
| **Output Type** | `Dict[str, np.ndarray]` |
| **Output Keys** | `'FFC'`, `'GC'`, `'WB'`, `'CC'` |
| **Output Shape** | `(H, W, 3)` - same as input |
| **Output dtype** | `float64` |
| **Output Range** | `[0, 1]` - clipped |
| **Output Color** | RGB |

---

## Related Documentation

- **Full Analysis**: See `predict_image_analysis.md` for detailed documentation
- **Quick Reference**: See `predict_image_quick_reference.md` for quick lookup
- **Test Suite**: See `tests/test_predict_image.py` for validation tests
- **Main README**: See repository README.md for package overview

---

## Questions & Answers

**Q: What if I only want the final corrected image?**  
A: Access `results['CC']`, but check if it's not `None` first.

**Q: Can I use this without training models?**  
A: No, you must train models first using `run()` or load saved models.

**Q: What if one of the correction stages fails?**  
A: The pipeline continues, and that stage's output will be the same as its input.

**Q: How do I convert the output back to display/save?**  
A: Convert: `(results['CC'][:, :, ::-1] * 255).astype('uint8')` for OpenCV/saving.

**Q: Can I modify the correction strength?**  
A: Not directly in `predict_image`. Adjust training parameters in `run()` instead.

**Q: What's the difference between 'CC' being None vs an array?**  
A: `None` means no color correction model was trained, array means model exists and was applied.

---

*Generated from analysis of ColorCorrectionPipeline/ccp.py (lines 622-704)*
