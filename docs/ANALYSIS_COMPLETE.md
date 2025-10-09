# Analysis Complete: `predict_image` Method

## Task Summary

**Objective:** Analyse the `predict_image` call in the ColorCorrectionPackage to document its inputs and outputs.

**Status:** ✅ COMPLETE

---

## Analysis Results

### Method Location
- **File:** `ColorCorrectionPipeline/ccp.py`
- **Class:** `ColorCorrection`
- **Method:** `predict_image`
- **Lines:** 622-704

---

## INPUTS - Detailed Analysis

### Input 1: `Image` (Required Parameter)

**Type:** `Union[str, np.ndarray]`

**Description:** The input image to be color corrected

**Accepted Values:**

#### Option A: File Path (String)
```python
Image: str = "path/to/image.jpg"
```
- **Format:** Any file format supported by OpenCV (JPEG, PNG, TIFF, BMP, etc.)
- **Processing:** 
  - Loaded using `cv2.imread()`
  - Automatically converted from BGR to RGB
  - Scaled from `uint8 [0, 255]` to `float64 [0, 1]`
- **Example:**
  ```python
  results = cc.predict_image(Image="test.jpg")
  ```

#### Option B: NumPy Array
```python
Image: np.ndarray
```
- **Shape:** `(height, width, 3)`
- **dtype:** `float64`
- **Range:** `[0, 1]`
- **Color Order:** RGB (NOT BGR)
- **Example:**
  ```python
  img_rgb = np.random.rand(100, 100, 3).astype(np.float64)
  results = cc.predict_image(Image=img_rgb)
  ```

**Error Handling:**
- Raises `FileNotFoundError` if file path cannot be read
- Raises `TypeError` if input is neither string nor numpy array

---

### Input 2: `show` (Optional Parameter)

**Type:** `bool`

**Default:** `False`

**Description:** Controls visualization of correction stages

**Values:**
- `True`: Displays plots for each correction stage using `colour.plotting.plot_image()`
- `False`: Silent operation, no visualization

**Example:**
```python
# With visualization
results = cc.predict_image(Image="test.jpg", show=True)

# Without visualization (default)
results = cc.predict_image(Image="test.jpg", show=False)
results = cc.predict_image(Image="test.jpg")  # Same as above
```

**Note:** Visualization errors are caught and silently ignored to prevent pipeline failure.

---

## OUTPUTS - Detailed Analysis

### Return Type
```python
Dict[str, np.ndarray]
```

### Dictionary Structure

The method returns a dictionary with exactly 4 keys:

```python
{
    'FFC': np.ndarray,        # Flat-Field Corrected
    'GC':  np.ndarray,        # Gamma Corrected
    'WB':  np.ndarray,        # White Balanced
    'CC':  np.ndarray | None  # Color Corrected (final)
}
```

---

### Output 1: `results['FFC']` - Flat-Field Corrected Image

**Type:** `np.ndarray`

**Properties:**
- **Shape:** `(height, width, 3)` - same as input
- **dtype:** `float64`
- **Range:** `[0, 1]`
- **Color Space:** RGB

**Description:** Image after flat-field correction, which compensates for uneven illumination across the image field.

**Availability:** ALWAYS present

**Behavior:**
- If `model_ffc` exists: Applies flat-field correction using the trained multiplier
- If `model_ffc` is `None`: Returns the original input image unchanged

**Use Case:** Corrects vignetting and uneven lighting patterns

---

### Output 2: `results['GC']` - Gamma Corrected Image

**Type:** `np.ndarray`

**Properties:**
- **Shape:** `(height, width, 3)` - same as input
- **dtype:** `float64`
- **Range:** `[0, 1]`
- **Color Space:** RGB

**Description:** Image after gamma correction, which adjusts brightness and contrast using a polynomial mapping of luminance values.

**Availability:** ALWAYS present

**Behavior:**
- If `model_gc` exists: Applies polynomial gamma correction to the L* channel in CIELAB space
- If `model_gc` is `None`: Returns the FFC output unchanged

**Use Case:** Corrects non-linear brightness response and improves contrast

---

### Output 3: `results['WB']` - White Balanced Image

**Type:** `np.ndarray`

**Properties:**
- **Shape:** `(height, width, 3)` - same as input
- **dtype:** `float64`
- **Range:** `[0, 1]`
- **Color Space:** RGB

**Description:** Image after white balance correction, which adjusts color temperature using a diagonal matrix transformation.

**Availability:** ALWAYS present

**Behavior:**
- If `model_wb` exists: Applies diagonal white balance matrix via matrix multiplication
- If `model_wb` is `None`: Returns the GC output unchanged

**Use Case:** Corrects color cast due to illumination color temperature

---

### Output 4: `results['CC']` - Color Corrected Image (FINAL)

**Type:** `np.ndarray` OR `None`

**Properties (when not None):**
- **Shape:** `(height, width, 3)` - same as input
- **dtype:** `float64`
- **Range:** `[0, 1]`
- **Color Space:** RGB

**Description:** Final color-corrected image using either conventional or ML-based color mapping.

**Availability:** ALWAYS present as a key, but VALUE may be `None`

**Behavior:**
- If `model_cc` exists with `'conv'` method: Applies Finlayson 2015 color correction matrix
- If `model_cc` exists with `'ours'` method: Applies custom ML model (linear, PLS, or neural network)
- If `model_cc` is `None`: Returns `None`

**Use Case:** Final color mapping to match reference colors (ColorChecker)

**Important:** This is the final output and typically the desired result. Always check if it's not `None` before using.

---

## Processing Pipeline Flow

The corrections are applied sequentially:

```
Input Image
    ↓
[1] Flat-Field Correction (FFC)
    ├─ Corrects: Uneven illumination
    ├─ Model: self.models.model_ffc
    └─ Output: results['FFC']
    ↓
[2] Gamma Correction (GC)
    ├─ Corrects: Brightness/contrast
    ├─ Model: self.models.model_gc
    └─ Output: results['GC']
    ↓
[3] White Balance (WB)
    ├─ Corrects: Color temperature
    ├─ Model: self.models.model_wb
    └─ Output: results['WB']
    ↓
[4] Color Correction (CC)
    ├─ Corrects: Color mapping
    ├─ Model: self.models.model_cc
    └─ Output: results['CC']
    ↓
Dictionary Output
```

---

## Complete Working Example

```python
import numpy as np
import cv2
from ColorCorrectionPipeline.ccp import ColorCorrection
from ColorCorrectionPipeline.Configs.configs import Config
from ColorCorrectionPipeline.key_functions import to_float64

# ============================================================
# PHASE 1: Train Models (one-time setup)
# ============================================================

cc = ColorCorrection()

config = Config(
    do_ffc=True,
    do_gc=True,
    do_wb=True,
    do_cc=True
)

# Train on image with color checker
metrics, images, errors = cc.run(
    Image="training_image.jpg",
    White_Image="white_image.jpg",  # Optional
    config=config
)

# ============================================================
# PHASE 2: Apply to New Images
# ============================================================

# Method A: Using file path
results = cc.predict_image(
    Image="new_image.jpg",
    show=False
)

# Method B: Using numpy array
img_bgr = cv2.imread("new_image.jpg")
img_rgb = to_float64(img_bgr[:, :, ::-1])
results = cc.predict_image(
    Image=img_rgb,
    show=False
)

# ============================================================
# PHASE 3: Access Results
# ============================================================

# Get individual stages
ffc_image = results['FFC']  # After flat-field correction
gc_image = results['GC']    # After gamma correction
wb_image = results['WB']    # After white balance
cc_image = results['CC']    # Final corrected (or None)

# Save final result
if cc_image is not None:
    output_bgr = (cc_image[:, :, ::-1] * 255).astype('uint8')
    cv2.imwrite("corrected.jpg", output_bgr)
```

---

## Input/Output Comparison Table

| Aspect | Input | Output |
|--------|-------|--------|
| **Parameter Name** | `Image` | Return value |
| **Type** | `str` OR `np.ndarray` | `Dict[str, np.ndarray]` |
| **Image Type** | File path or array | Dictionary of arrays |
| **Shape** | `(H, W, 3)` | Each array: `(H, W, 3)` |
| **dtype** | `float64` (if array) | All arrays: `float64` |
| **Range** | `[0, 1]` (if array) | All arrays: `[0, 1]` |
| **Color Space** | RGB | All arrays: RGB |
| **Keys** | N/A | `'FFC'`, `'GC'`, `'WB'`, `'CC'` |

---

## Key Findings

### What the Method Does
1. ✅ Applies sequential color correction pipeline
2. ✅ Uses pre-trained models (from `run()` method)
3. ✅ Works on any image (no color checker needed)
4. ✅ Returns intermediate results from each stage
5. ✅ Automatically handles color space conversions
6. ✅ Clips all outputs to valid [0,1] range
7. ✅ Logs processing time

### What the Method Requires
1. ⚠️ Models must be trained first using `run()`
2. ⚠️ Reference illuminant must be set (done during training)
3. ⚠️ Input must be RGB format (not BGR)
4. ⚠️ Input array must be float64 in [0,1] range

### What the Method Returns
1. ✅ Dictionary with 4 keys: `'FFC'`, `'GC'`, `'WB'`, `'CC'`
2. ✅ All values are `np.ndarray` except `'CC'` which can be `None`
3. ✅ All arrays have same shape as input
4. ✅ All arrays are RGB, float64, [0,1] range

---

## Documentation Files Created

1. **[predict_image_summary.md](./predict_image_summary.md)**
   - Executive summary with complete examples
   - Q&A section
   - Best for: Quick start

2. **[predict_image_analysis.md](./predict_image_analysis.md)**
   - Comprehensive technical documentation
   - Detailed parameter specifications
   - Best for: Deep understanding

3. **[predict_image_quick_reference.md](./predict_image_quick_reference.md)**
   - Quick reference card
   - Tables and condensed information
   - Best for: Quick lookup while coding

4. **[predict_image_diagram.txt](./predict_image_diagram.txt)**
   - Visual ASCII diagrams
   - Pipeline flow visualization
   - Best for: Understanding structure

5. **[README.md](./README.md)**
   - Documentation index
   - Navigation guide
   - Best for: Finding documentation

6. **[ANALYSIS_COMPLETE.md](./ANALYSIS_COMPLETE.md)** (this file)
   - Analysis summary
   - Complete findings
   - Best for: Task completion review

7. **[../tests/test_predict_image.py](../tests/test_predict_image.py)**
   - Test suite validating documentation
   - Input/output validation
   - Best for: Verification

---

## Conclusion

The `predict_image` method in the ColorCorrectionPipeline package has been thoroughly analyzed and documented:

### Inputs Summary
- **Parameter 1 (Image):** File path (string) OR RGB numpy array (float64, [0,1])
- **Parameter 2 (show):** Boolean flag for visualization (default: False)

### Outputs Summary
- **Return Type:** Dictionary with 4 keys
- **Keys:** `'FFC'`, `'GC'`, `'WB'`, `'CC'`
- **Values:** NumPy arrays (RGB, float64, [0,1]) or `None` for `'CC'`
- **All outputs:** Same shape as input, clipped to valid range

### Pipeline Summary
Sequential correction: Input → FFC → GC → WB → CC → Output Dictionary

The analysis is complete, comprehensive, and validated with test cases.

---

**Analysis Date:** 2024-10-09  
**Source Code:** ColorCorrectionPipeline/ccp.py (lines 622-704)  
**Package Version:** 1.2.01+
