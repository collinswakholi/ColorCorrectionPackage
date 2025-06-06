# ColorCorrectionPipeline

A step-wise, end-to-end color‐correction pipeline for digital images.  
This package combines flat-field correction (**FFC**), gamma correction (**GC**), white-balance (**WB**), and color-correction (**CC**) into a single, easy-to-use workflow. Once you “train” on an image containing a color-checker (and a white-field for FFC), you can apply the learned corrections to any new image (no chart required, as long as it was captured with the same camera, and in the same lighting conditions).

This package builds upon a previous package [ML_ColorCorrection_tool](https://github.com/collinswakholi/ML_ColorCorrection_tool) package.

---
## Features

- **Flat-Field Correction (FFC)**  
  Automatically detect or manually crops “white” background image. Fits an n-degree 2D surface to describe the light distribution in the FOV, extrapolates to full image.

- **Saturation Check / Extrapolation**  
  Identify and fix saturated patches on the chart before proceeding, ensuring accurate downstream corrections.

- **Gamma Correction (GC)**  
  Fits an optimum polynomial (up to configurable degree) mapping between measured neutral patch intensities and reference values, and applies it to the entire image.

- **White Balance (WB)**  
  Diagonal white-balance correction using the neutral patches of the color checker. Gets diagonal matrix and applies it to the entire image.

- **Color Correction (CC)**  
  Two methods:
  - **Conventional (“conv”)**: configurable polynomial expansion with the Finlayson 2015 method, produces a 3xn matrix that can be applied to the entire image.
  - **Custom (“ours”)**: uses ML with linear regression, pls regression, or neural networks, produces a model that can be applied to the entire image.

- **Predict on New Images**  
  Once models are saved, apply FFC → GC → WB → CC in sequence to any new photograph, no chart needed.

---
## Installation

### From PyPI

```bash
pip install ColorCorrectionPipeline
```

### From GitHub

```bash
git clone https://github.com/collinswakholi/ColorCorrectionPipeline.git
cd ColorCorrectionPipeline
pip install -e .
```

### Dependencies
The Dependencies (Automatically Installed, from `requirements.txt`) are:
- `numpy`
- `scipy`
- `scikit-learn`
- `torch`
- `opencv-python`
- `colour-science`
- `color-checker-detection`
- `ultralytics`
- `scikit-image`
- `plotly`
- `matplotlib`
- `pandas`
- `difflib`
- `glob2`
- `statsmodels`
- `seaborn` 
- `pytest`

If you already have a `requirements.txt` file in your cloned repository, you can install them using `pip install -r requirements.txt`.

---
## Usage
Below is a simple example of how to use the package (found in `quick_run.py`):
```python
import os
import cv2
import numpy as np
import pandas as pd

from ColorCorrectionPipeline.core import ColorCorrection, Config
from ColorCorrectionPipeline.key_functions import to_float64

# ─────────────────────────────────────────────────────────────────────────────
# 1. File paths
# ─────────────────────────────────────────────────────────────────────────────
IMG_PATH         = "Data/Sample_1.JPG"        # Image containing color checker
WHITE_PATH       = "Data/white.JPG"           # Optional White background image for FFC
YOLO_MODEL_PATH  = "Data/plane_det_model_YOLO_512_n.pt"  # Optional YOLO .pt
TEST_IMAGE_PATH  = "Data/Sample_2.JPG"        # Optional New image for prediction

# Output directory (only used if config.save=True)
SAVE_PATH = os.path.join(os.getcwd(), "results")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load images and convert to RGB float64 in [0,1]
# ─────────────────────────────────────────────────────────────────────────────
img_bgr   = cv2.imread(IMG_PATH)
img_rgb   = to_float64(img_bgr[:, :, ::-1])# convert to RGB (64bit floats, 0-1, RGB)

white_bgr = cv2.imread(WHITE_PATH)

test_bgr  = cv2.imread(TEST_IMAGE_PATH)
test_rgb  = to_float64(test_bgr[:, :, ::-1])# convert to RGB (64bit floats, 0-1, RGB)

img_name = os.path.splitext(os.path.basename(IMG_PATH))[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Configure per‐stage parameters
# ─────────────────────────────────────────────────────────────────────────────

ffc_kwargs = {
    "model_path": YOLO_MODEL_PATH, # Optional, for automatic white plane ROI detection
    "manual_crop": False, # Optional, for manual white plane ROI selection
    "show": False, # Whether to show intermediate plots
    "bins": 50, # Number of bins used for sampling the intesity profile of the white plane
    "smooth_window": 5, # Window size for smoothing the intensity profile
    "get_deltaE": True, # Whether to calculate and return deltaE (CIEDE2000)
    "fit_method": "pls", # can be linear, nn, pls, or svm, default is linear
    "interactions": True, # Whether to include interactions in the polynomial expansion
    "max_iter": 1000, # Maximum number of iterations
    "tol": 1e-8, # Tolerance for stopping criterion
    "verbose": False, # Whether to print verbose output
    "random_seed": 0, # Random seed
}

# Gamma Correction (GC) kwargs:
gc_kwargs = {
    "max_degree": 5, # Maximum polynomial degree for fitting gamma profile
    "show": False, # Whether to show intermediate plots
    "get_deltaE": True, # Whether to calculate and return deltaE (CIEDE2000)
}

# White Balance (WB) kwargs:
wb_kwargs = {
    "show": False, # Whether to show intermediate plots
    "get_deltaE": True, # Whether to calculate and return deltaE (CIEDE2000)
}

# Color Correction (CC) kwargs:
cc_kwargs = {
    'cc_method': 'ours', # method to use for color correction
    'method': 'Finlayson 2015', # if cc_method is 'conv', this is the method
    'mtd': 'nn', # if cc_method is 'ours', this is the method, linear, nn, pls

    'degree': 2, # degree of polynomial to fit
    'max_iterations': 10000, # max iterations for fitting
    'random_state': 0, # random seed
    'tol': 1e-8, # tolerance for fitting
    'verbose': False, # whether to print verbose output
    'param_search': False, # whether to use parameter search
    'show': False, # whether to show plots
    'get_deltaE': True, # whether to compute deltaE
    'n_samples': 50, # number of samples to use for parameter search

    # only if mtd == 'pls'
    'ncomp': 1, # number of components to use

    # only if mtd == 'nn'
    'nlayers': 100, # number of layers to use
    'hidden_layers': [64, 32, 16], # hidden layers for neural network
    'learning_rate': 0.001, # learning rate for neural network
    'batch_size': 16, # batch size for neural network
    'patience': 10, # patience for early stopping
    'dropout_rate': 0.2, # dropout rate for neural network
    'optim_type': 'adam', # optimizer type for neural network
    'use_batch_norm': True, # whether to use batch normalization
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Config and run the Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
config = Config(
    do_ffc=True, # Change to False if you don't want to run FFC
    do_gc=True, # Change to False if you don't want to run GC
    do_wb=True, # Change to False if you don't want to run WB
    do_cc=True, # Change to False if you don't want to run CC
    save=False,            # Change to True if you want to save models + CSVs
    save_path=SAVE_PATH,   # Directory for saving outputs (models & CSV)
    check_saturation=True, # Change to False if you don't want to check if color chart patches are saturated
    REF_ILLUMINANT=None,   # Defaults to D65; supply np.ndarray if needed
    FFC_kwargs=ffc_kwargs,
    GC_kwargs=gc_kwargs,
    WB_kwargs=wb_kwargs,
    CC_kwargs=cc_kwargs,
)

cc = ColorCorrection() # Initialize ColorCorrection class
metrics, corrected_imgs, errors = cc.run(
    Image=img_rgb,
    White_Image=white_bgr, # Optional, you do have to pass anything
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
```

### Assuming you have;
1. A photograph with a color checker chart: `Data/Sample_1.JPG`, 
2. An optional matching white-field image (for FFC): `Data/white.JPG`,
3. A YOLO model for detecting the white plane (optional if you want automatic ROI): `Data/plane_det_model_YOLO_512_n.pt`
4. Another optional image (no chart required) to test the learned corrections: `Data/Sample_2.JPG`

## Sample Reusults
Before color correction:
![Before](Images/before.svg)

Same images after color correction:
![After](Images/after.svg)

## References
A detailed study that led to this package can be found at: [Awaiting Publication](https://www.yet_to_publish.com).

key packages used: 
- Colour-science package: [https://colour-science.org](https://colour-science.org)
- scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- opencv-python: [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/)


## Contributions
- [Collins Wakholi](https://github.com/collinswakholi)
- [Devin A. Rippner](https://github.com/daripp)

## Acknowledgements
I am gratefully acknowledge [Devin A. Rippner](https://github.com/daripp), [ORISE](https://orise.orau.gov/index.html), and the [USDA-ARS](https://www.ars.usda.gov) for their invaluable assistance and funding support in the development of this Repo. This project would not have been possible without their guidance and opportunities provided.