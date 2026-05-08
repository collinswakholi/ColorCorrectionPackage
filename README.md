# ColorCorrectionPipeline

A comprehensive, step-by-step color correction pipeline for digital images. This package integrates flat-field correction (FFC), gamma correction (GC), white balance (WB), and color correction (CC) into a unified, user-friendly workflow. After training on a reference image with a color checker chart (and optionally a white-field image for FFC), the learned corrections can be applied to any new image captured under the same conditions—no color chart required for subsequent images.

**This package builds upon a previous package _[ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool)_.**

**A UI version of this package can be found at _[ColorCorrectionPackage_UI](https://github.com/collinswakholi/ColorCorrectionPackage_UI)_.**

## Associated Manuscript

This package implements the open-source color-correction workflow described in:

Wakholi, C., Hardigan, M. A., Lee, J., Lukas, S. B., Feldman, M. J., Altendorf, K. R., Neyhart, J. L., & Rippner, D. A. (2026). A systematic color correction pipeline for controlled-environment imaging. *The Plant Phenome Journal*, 9(1), e70067. https://doi.org/10.1002/ppj2.70067

Manuscript link: https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/ppj2.70067

## Features

• **Flat-Field Correction (FFC)**  
  Automatically detect or manually crop the "white" background image. Automatic plane detection uses the bundled ONNX model through OpenCV DNN, so no `ultralytics` runtime dependency is required. FFC multipliers are cached per white image and configuration to speed repeated runs.

• **Saturation Check / Extrapolation**  
  Identify and fix saturated patches on the chart before proceeding, ensuring accurate downstream corrections.

• **Gamma Correction (GC)**  
  Fits an optimum polynomial (up to configurable degree) mapping between measured neutral patch intensities and reference values, and applies it to the entire image.

• **White Balance (WB)**  
  Diagonal white-balance correction using the neutral patches of the color checker. Gets diagonal matrix and applies it to the entire image.

• **Color Correction (CC)**  
  Two methods:
  - Conventional ("conv"): configurable polynomial expansion with the Finlayson 2015 method, produces a 3xn matrix that can be applied to the entire image.
  - Custom ("ours"): uses ML with linear regression, PLS regression, or neural networks, produces a model that can be applied to the entire image. For real-image NN workflows, `mtd="nn"`, `degree=2`, `n_samples=50`, and `hidden_layers=[64]` is the recommended starting point.

• **Predict on New Images**  
  Once models are saved, apply FFC → GC → WB → CC in sequence to any new photograph, no chart needed.

• **⚡ Hardware Acceleration (Numba/CUDA)**  
  Automatic detection of CPU parallelism and CUDA at import time. Numba-JIT kernels accelerate sRGB↔Lab conversion (via precomputed LUTs), 3-D LUT trilinear interpolation for CC prediction, and FFC. CC LUTs are built lazily by default so small chart-metric predictions do not pay the full LUT startup cost.

• **📦 Batch Prediction (`predict_images()`)**  
  Apply the full FFC → GC → WB → CC pipeline to a list of images in parallel using a `ThreadPoolExecutor`. Accepts file paths or pre-loaded arrays, and an optional `on_progress` callback for real-time progress tracking.

## Package Structure

The ColorCorrectionPipeline package includes the following key components:

```
ColorCorrectionPipeline/
├── __init__.py               # Package exports
├── __version__.py            # Version information
├── pipeline.py               # Main ColorCorrection class
├── models.py                 # Model definitions and persistence
├── config.py                 # Configuration management
├── constants.py              # Package constants
├── core/                     # Core algorithms
│   ├── __init__.py
│   ├── accel.py              # Hardware acceleration (Numba CPU/CUDA kernels)
│   ├── color_spaces.py       # Color space conversions
│   ├── correction.py         # Correction algorithms
│   ├── metrics.py            # Quality metrics (ΔE)
│   ├── transforms.py         # Image transformations
│   └── utils.py              # Utility functions
├── flat_field/               # Flat-Field Correction module
│   ├── __init__.py
│   ├── correction.py         # FFC implementation
│   └── models/               # Pre-trained models (included in package)
│       ├── __init__.py
│       └── plane_det_model_YOLO_512_n.onnx  # OpenCV DNN model for automatic white plane detection
└── io/                       # I/O utilities
    ├── __init__.py
    ├── readers.py            # Image readers
    └── writers.py            # Image writers
```

Note: The ONNX plane detection model (`plane_det_model_YOLO_512_n.onnx`) is automatically included when you install the package, so you don't need to download or specify the model path separately.

## Release Highlights

### 1.4.5

- Replaced runtime `ultralytics` plane detection with OpenCV DNN inference using the bundled ONNX detector.
- Added flat-field multiplier caching for repeated white-image/configuration pairs.
- Made color-correction 3-D LUT construction lazy and configurable with `use_lut`, `lazy_lut`, `lut_grid_size`, and `lut_min_pixels`.
- Hardened custom PyTorch NN training around batch normalization and CUDA out-of-memory fallback behavior.
- Improved `n_samples > 1` color-correction training by avoiding duplicate chart detection.
- Updated the recommended sklearn NN color-correction starting point to `hidden_layers=[64]`, which gave the best speed/accuracy balance on the bundled real-image test.

## Installation

### Quick Start (Recommended)

Install directly from PyPI:

```bash
pip install ColorCorrectionPipeline
```

`numba` is installed automatically. Hardware acceleration (CPU parallelism and CUDA, if an NVIDIA GPU is present) is detected and enabled at import time — no extra install flags or code changes needed.

### Development Installation

For the latest features or development:

```bash
# Clone the repository
git clone https://github.com/collinswakholi/ColorCorrectionPackage.git
cd ColorCorrectionPackage

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Requirements

• Python: 3.8 or higher  
• Operating System: Windows, macOS, Linux  
• Memory: Minimum 4GB RAM (8GB recommended for large images)  
• GPU: Optional (CUDA-compatible GPU for accelerated processing)

### Dependencies

The package automatically installs the following dependencies:

**Core Dependencies:**
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning algorithms
- `opencv-contrib-python` - Computer vision with MCC color-checker detection
- `torch` - Deep learning framework
- `numba` - JIT-compiled CPU/CUDA kernels for accelerated image processing

**Image Processing:**
• `scikit-image` - Image processing algorithms  
• `colour-science` - Color science computations  
• `colour-checker-detection` - Optional segmentation-based color checker sampling (`pip install "ColorCorrectionPipeline[segmentation]"`)

**Visualization & Analysis:**
• `matplotlib`, `plotly`, `seaborn` - Plotting and visualization  
• `pandas` - Data manipulation  
• `statsmodels` - Statistical modeling

**Development & Testing:**
• `pytest` - Testing framework

### Verify Installation

Verify your installation:

```python
import ColorCorrectionPipeline
from ColorCorrectionPipeline import ColorCorrection
```

## Usage

Below is a simple example of how to use the package:

```python
import os
import cv2
import numpy as np
import pandas as pd

from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.core.utils import to_float64

# ─────────────────────────────────────────────────────────────────────────────
# 1. File paths
# ─────────────────────────────────────────────────────────────────────────────
IMG_PATH         = "Images/Sample_1.JPG"        # Image containing color checker
WHITE_PATH       = "Images/white.JPG"           # Optional White background image for FFC
TEST_IMAGE_PATH  = "Images/Image_1.JPG"         # Optional New image for prediction

# Output directory (only used if config.save=True)
SAVE_PATH = os.path.join(os.getcwd(), "results")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load images and convert to RGB float64 in [0,1]
# ─────────────────────────────────────────────────────────────────────────────
img_bgr   = cv2.imread(IMG_PATH)
img_rgb   = to_float64(img_bgr[:, :, ::-1])  # convert to RGB (64bit floats, 0-1, RGB)

white_bgr = cv2.imread(WHITE_PATH)

test_bgr  = cv2.imread(TEST_IMAGE_PATH)
test_rgb  = to_float64(test_bgr[:, :, ::-1])  # convert to RGB (64bit floats, 0-1, RGB)

img_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Configure per‐stage parameters
# ─────────────────────────────────────────────────────────────────────────────

ffc_kwargs = {
    "manual_crop": False,           # Optional, for manual white plane ROI selection
    "show": False,                  # Whether to show intermediate plots
    "bins": 50,                     # Number of bins used for sampling the intensity profile of the white plane
    "smooth_window": 5,             # Window size for smoothing the intensity profile
    "get_deltaE": True,             # Whether to calculate and return deltaE (CIEDE2000)
    "fit_method": "pls",            # can be linear, nn, pls, or svm, default is linear
    "interactions": True,           # Whether to include interactions in the polynomial expansion
    "max_iter": 1000,               # Maximum number of iterations
    "tol": 1e-8,                    # Tolerance for stopping criterion
    "verbose": False,               # Whether to print verbose output
    "random_seed": 0,               # Random seed
    "cache_multiplier": True,       # Reuse multiplier for same white image/config
}

# Gamma Correction (GC) kwargs:
gc_kwargs = {
    "max_degree": 5,                # Maximum polynomial degree for fitting gamma profile
    "show": False,                  # Whether to show intermediate plots
    "get_deltaE": True,             # Whether to calculate and return deltaE (CIEDE2000)
}

# White Balance (WB) kwargs:
wb_kwargs = {
    "show": False,                  # Whether to show intermediate plots
    "get_deltaE": True,             # Whether to calculate and return deltaE (CIEDE2000)
}

# Color Correction (CC) kwargs:
cc_kwargs = {
    'cc_method': 'ours',            # method to use for color correction
    'method': 'Finlayson 2015',     # if cc_method is 'conv', this is the method
    'mtd': 'nn',                    # if cc_method is 'ours', this is the method, linear, nn, pls

    'degree': 2,                    # degree of polynomial to fit
    'max_iterations': 10000,        # max iterations for fitting
    'random_state': 0,              # random seed
    'tol': 1e-8,                    # tolerance for fitting
    'verbose': False,               # whether to print verbose output
    'param_search': False,          # whether to use parameter search
    'show': False,                  # whether to show plots
    'get_deltaE': True,             # whether to compute deltaE
    'n_samples': 50,                # number of samples to use for parameter search

    # only if mtd == 'pls' otherwise disable
    # 'ncomp': 1,                     # number of components to use

    # only if mtd == 'nn' or mtd == 'custom' otherwise disable
    'hidden_layers': [64],          # recommended hidden layer size for sklearn NN
    'learning_rate': 0.001,         # learning rate for neural network
    'batch_size': 16,               # batch size for neural network
    'patience': 10,                 # patience for early stopping
    'dropout_rate': 0.2,            # dropout rate for neural network
    'optim_type': 'adam',           # optimizer type for neural network
    'use_batch_norm': False,        # only used by mtd == 'custom'
    'use_lut': True,                # use 3-D LUT acceleration for image prediction
    'lazy_lut': True,               # build LUT only when first needed
    'lut_grid_size': 33,            # 3-D LUT grid resolution
    'lut_min_pixels': 4096,         # direct-predict small arrays below this size
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Config and run the Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
config = Config(
    do_ffc=True,                    # Change to False if you don't want to run FFC
    do_gc=True,                     # Change to False if you don't want to run GC
    do_wb=True,                     # Change to False if you don't want to run WB
    do_cc=True,                     # Change to False if you don't want to run CC
    save=False,                     # Change to True if you want to save models + CSVs
    save_path=SAVE_PATH,            # Directory for saving outputs (models & CSV)
    check_saturation=True,          # Change to False if you don't want to check if color chart patches are saturated
    REF_ILLUMINANT=None,            # Defaults to D65; supply np.ndarray if needed
    FFC_kwargs=ffc_kwargs,
    GC_kwargs=gc_kwargs,
    WB_kwargs=wb_kwargs,
    CC_kwargs=cc_kwargs,
)

cc = ColorCorrection()              # Initialize ColorCorrection class
metrics, corrected_imgs, errors = cc.run(
    Image=img_rgb,
    White_Image=white_bgr,          # Optional, you don't have to pass anything
    name_=img_name,
    config=config,
)

# Convert metrics (dict) → pandas.DataFrame for display
metrics_df = pd.DataFrame.from_dict(metrics)
print("Per-patch and summary metrics for each stage:\n", metrics_df.head())

# ─────────────────────────────────────────────────────────────────────────────
# 5. Predict on a New Image (no color-checker required)
# ─────────────────────────────────────────────────────────────────────────────
test_results = cc.predict_image(test_rgb, show=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch-predict multiple images in parallel
# ─────────────────────────────────────────────────────────────────────────────
def on_progress(done, total, name):
    print(f"[{done}/{total}] finished: {name}")

batch_results = cc.predict_images(
    images=["Images/Image_1.JPG"],
    max_workers=4,
    on_progress=on_progress,
)
# batch_results is a list of dicts, one per image, with keys: FFC, GC, WB, CC
```

### Assuming you have;

1. A photograph with a color checker chart: `Images/Sample_1.JPG`,
2. An optional matching white-field image (for FFC): `Images/white.JPG`,
3. The ONNX model for detecting the white plane is now automatically included in the package: `ColorCorrectionPipeline/flat_field/models/plane_det_model_YOLO_512_n.onnx`
4. Another optional image (no chart required) to test the learned corrections: `Images/Image_1.JPG`

## Sample Results

The ColorCorrectionPipeline delivers significant improvements in color accuracy and consistency. Below are sample results demonstrating the effectiveness of the complete correction pipeline:

### Before Color Correction

Raw images straight from the camera showing color cast, vignetting, and inconsistent color reproduction:

![Before Color Correction](ReadMe_Images/before.svg)

### After Color Correction

Same images after applying the complete FFC → GC → WB → CC pipeline, showing improved color accuracy, uniform illumination, and consistent color reproduction:

![After Color Correction](ReadMe_Images/After.svg)

**Key Improvements:**

• ✅ Eliminated vignetting and illumination non-uniformities (FFC)  
• ✅ Corrected gamma response for accurate neutral tones (GC)  
• ✅ Achieved neutral white balance under the reference illuminant (WB)  
• ✅ Accurate color reproduction matching reference standards (CC)  
• ✅ Consistent results across multiple images captured under the same conditions

Typical results after full pipeline correction achieve ΔE < 2.0 for most images, with many achieving ΔE < 1.2.

## Contributing

We welcome contributions! Please see our contributing guidelines below:

1. **Fork and Clone**

```bash
git clone https://github.com/collinswakholi/ColorCorrectionPackage.git
cd ColorCorrectionPackage
```

2. **Create Development Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. **Run Tests**

```bash
pytest tests/
```

4. **Code Style**

```bash
black .
```

5. **Submit a Pull Request**

### License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/collinswakholi/ColorCorrectionPackage/blob/main/LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

This package accompanies the manuscript [A systematic color correction pipeline for controlled-environment imaging](https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/ppj2.70067), published in *The Plant Phenome Journal*.

```bibtex
@software{colorcorrectionpipeline,
    author = {Wakholi, Collins and Rippner, Devin A.},
    title = {ColorCorrectionPipeline: A stepwise color‐correction pipeline},
    url = {https://github.com/collinswakholi/ColorCorrectionPackage},
    version = {1.4.5},
    year = {2026}
}

@article{wakholi2026systematic,
  author = {Wakholi, Collins and Hardigan, Michael A. and Lee, Jungmin and Lukas, Scott B. and Feldman, Max J. and Altendorf, Kayla R. and Neyhart, Jeffrey L. and Rippner, Devin A.},
  title = {A systematic color correction pipeline for controlled-environment imaging},
  journal = {The Plant Phenome Journal},
  volume = {9},
  number = {1},
  pages = {e70067},
  year = {2026},
  doi = {10.1002/ppj2.70067},
  url = {https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/ppj2.70067}
}
```

## Acknowledgements

We would like to gratefully acknowledge:

• [Devin A. Rippner](https://github.com/daripp) for invaluable technical guidance  
• [ORISE](https://orise.orau.gov/index.html) for fellowship support  
• [USDA-ARS](https://www.ars.usda.gov/) for funding and research opportunities

Made with ❤️ by Collins Wakholi

For bug reports and feature requests, please open an issue on [GitHub](https://github.com/collinswakholi/ColorCorrectionPackage/issues).
