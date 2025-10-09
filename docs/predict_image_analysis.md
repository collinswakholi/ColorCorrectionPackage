# Analysis of `predict_image` Method

## Overview
The `predict_image` method in the `ColorCorrection` class applies pre-trained color correction models to a new image. This method is used after training the pipeline with the `run()` method and is designed to process images without requiring a color checker chart.

## Location
- **File**: `ColorCorrectionPipeline/ccp.py`
- **Class**: `ColorCorrection`
- **Method**: `predict_image`
- **Lines**: 622-704

## Method Signature

```python
def predict_image(
    self, 
    Image: Union[str, np.ndarray], 
    show: bool = False
) -> Dict[str, np.ndarray]:
```

## Input Parameters

### 1. `Image` (Required)
- **Type**: `Union[str, np.ndarray]`
- **Description**: The input image to be corrected
- **Accepted Formats**:
  - **String (file path)**: Path to an image file (e.g., `"path/to/image.jpg"`)
    - Supported formats: Any format supported by OpenCV's `cv2.imread()` (JPEG, PNG, TIFF, BMP, etc.)
    - The image is automatically loaded and converted from BGR to RGB
    - Pixel values are converted to float64 in the range [0, 1]
  - **NumPy Array**: Pre-loaded image as a numpy array
    - Expected shape: `(height, width, 3)` for RGB images
    - Expected dtype: float64 with values in range [0, 1]
    - Channel order: RGB (not BGR)
- **Raises**:
  - `FileNotFoundError`: If the image path cannot be read
  - `TypeError`: If the Image is neither a string nor a numpy array

### 2. `show` (Optional)
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to display visualization of the correction steps
- **Behavior**:
  - If `True`: Attempts to plot each intermediate result ('FFC', 'GC', 'WB', 'CC') using `colour.plotting.plot_image()`
  - If `False`: No visualization is shown
  - Note: Visualization failures are caught and silently ignored

## Output

### Return Value
- **Type**: `Dict[str, np.ndarray]`
- **Description**: A dictionary containing the results of each correction stage

### Dictionary Keys and Values

The returned dictionary contains the following keys:

#### 1. `'FFC'` - Flat-Field Corrected Image
- **Type**: `np.ndarray`
- **Shape**: `(height, width, 3)`
- **dtype**: `float64`
- **Range**: [0, 1]
- **Description**: Image after flat-field correction
- **When Available**: 
  - Always present in the output
  - If no FFC model was trained (`self.models.model_ffc is None`), this will be the original input image
  - If FFC model exists, this is the image after correcting for uneven illumination

#### 2. `'GC'` - Gamma Corrected Image
- **Type**: `np.ndarray`
- **Shape**: `(height, width, 3)`
- **dtype**: `float64`
- **Range**: [0, 1]
- **Description**: Image after gamma correction
- **When Available**:
  - Always present in the output
  - If no GC model was trained (`self.models.model_gc is None`), this will be the same as the FFC output
  - If GC model exists, this is the image after applying the polynomial gamma correction

#### 3. `'WB'` - White-Balanced Image
- **Type**: `np.ndarray`
- **Shape**: `(height, width, 3)`
- **dtype**: `float64`
- **Range**: [0, 1]
- **Description**: Image after white balance correction
- **When Available**:
  - Always present in the output
  - If no WB model was trained (`self.models.model_wb is None`), this will be the same as the GC output
  - If WB model exists, this is the image after applying the diagonal white balance matrix

#### 4. `'CC'` - Color Corrected Image
- **Type**: `np.ndarray` or `None`
- **Shape**: `(height, width, 3)` (when not None)
- **dtype**: `float64`
- **Range**: [0, 1]
- **Description**: Final color-corrected image
- **When Available**:
  - Present in the output (though may be `None`)
  - If no CC model was trained (`self.models.model_cc is None`), this will be `None`
  - If CC model exists, this is the final corrected image using either:
    - **'conv' method**: Conventional Finlayson 2015 color correction matrix
    - **'ours' method**: Custom ML-based color correction model

## Processing Pipeline

The method applies corrections sequentially:

```
Input Image → FFC → GC → WB → CC → Output Dictionary
```

Each stage:
1. Checks if the corresponding model exists in `self.models`
2. If model exists, applies the correction
3. If model doesn't exist, passes the image unchanged to the next stage
4. Stores the result in the output dictionary

## Timing Information

The method logs the total prediction time:
- Measures elapsed time from start to finish
- Logs to console: `"Prediction elapsed: {time:.2f}s"`
- Color: light green, style: italic, level: info

## Prerequisites

Before calling `predict_image`, you must:

1. **Create a ColorCorrection instance**:
   ```python
   cc = ColorCorrection()
   ```

2. **Train the models** using the `run()` method:
   ```python
   metrics, corrected_imgs, errors = cc.run(
       Image=training_image,
       White_Image=white_image,  # Optional
       name_="training",
       config=config
   )
   ```

3. **OR load pre-trained models**:
   ```python
   cc.models.load(model_path)
   ```

4. **Set reference values** (usually done automatically during training):
   ```python
   cc.get_reference_values(REF_ILLUMINANT=None)  # Uses D65 by default
   ```

## Example Usage

### Example 1: Basic Usage with File Path

```python
import cv2
from ColorCorrectionPipeline.ccp import ColorCorrection
from ColorCorrectionPipeline.Configs.configs import Config

# Train the pipeline
cc = ColorCorrection()
config = Config(do_ffc=True, do_gc=True, do_wb=True, do_cc=True)
metrics, images, errors = cc.run(
    Image="training_image.jpg",
    White_Image="white_image.jpg",
    config=config
)

# Apply to a new image
results = cc.predict_image(Image="new_image.jpg", show=False)

# Access individual correction stages
ffc_image = results['FFC']   # Flat-field corrected
gc_image = results['GC']      # Gamma corrected
wb_image = results['WB']      # White balanced
cc_image = results['CC']      # Final color corrected (or None)
```

### Example 2: Using NumPy Array Input

```python
import cv2
import numpy as np
from ColorCorrectionPipeline.key_functions import to_float64

# Load and prepare image
img_bgr = cv2.imread("new_image.jpg")
img_rgb = to_float64(img_bgr[:, :, ::-1])  # Convert BGR to RGB, scale to [0,1]

# Apply corrections
results = cc.predict_image(Image=img_rgb, show=False)

# Save final result
if results['CC'] is not None:
    final_image = (results['CC'] * 255).astype(np.uint8)
    cv2.imwrite("corrected.jpg", final_image[:, :, ::-1])  # Convert RGB to BGR
```

### Example 3: Visualizing All Stages

```python
# Show all correction stages
results = cc.predict_image(Image="new_image.jpg", show=True)

# This will attempt to display plots for each stage:
# - 'FFC': Flat-field corrected image
# - 'GC': Gamma corrected image
# - 'WB': White balanced image
# - 'CC': Final color corrected image (if model exists)
```

### Example 4: Processing Multiple Images

```python
import os
import glob

# Get all images in a directory
image_paths = glob.glob("test_images/*.jpg")

# Process each image
for img_path in image_paths:
    results = cc.predict_image(Image=img_path, show=False)
    
    # Save only the final corrected image
    if results['CC'] is not None:
        output_path = f"corrected_{os.path.basename(img_path)}"
        final_bgr = (results['CC'][:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(output_path, final_bgr)
```

## Important Notes

1. **Model Requirement**: The method requires that models have been trained or loaded. If no models exist, the output will contain the original image in various keys.

2. **Reference Illuminant**: The method uses `self.REF_ILLUMINANT` which should be set during training. If not set, some operations may fail.

3. **Memory Management**: 
   - After prediction, CUDA cache is cleared if GPU is used
   - Garbage collection is triggered automatically

4. **Color Space**: All input and output images are in RGB color space with float64 precision in range [0, 1]

5. **Clipping**: All intermediate results are clipped to [0, 1] range to ensure valid pixel values

6. **Error Handling**: Visualization errors (when `show=True`) are silently caught and ignored, allowing the method to complete even if plotting fails

## Performance Considerations

- **File I/O**: Loading from file path adds I/O overhead compared to using pre-loaded arrays
- **GPU Acceleration**: Gamma correction uses GPU if CUDA is available
- **Memory**: Each stage creates a copy of the image, so memory usage is proportional to image size × number of stages

## Related Methods

- **`run()`**: Main training method that creates the models used by `predict_image`
- **`get_reference_values()`**: Sets up reference color checker data
- **`models.save()`**: Save trained models for later use
- **`models.load()`**: Load pre-trained models

## See Also

- [ColorCorrection.run() documentation](link-to-run-docs)
- [Config class documentation](link-to-config-docs)
- [Model saving and loading](link-to-model-docs)
