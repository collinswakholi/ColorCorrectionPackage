# ColorCorrectionPipeline Documentation

This directory contains detailed documentation for the ColorCorrectionPipeline package.

## Available Documentation

### `predict_image` Method Documentation

Analysis of the `predict_image` method in the `ColorCorrection` class, which is used to apply trained color correction models to new images.

1. **[predict_image_summary.md](./predict_image_summary.md)** - Executive summary with complete examples
   - Quick overview of inputs and outputs
   - Complete working examples
   - Q&A section
   - Best for: Getting started quickly

2. **[predict_image_analysis.md](./predict_image_analysis.md)** - Detailed technical analysis
   - Comprehensive input parameter specifications
   - Detailed output structure documentation
   - Processing pipeline explanation
   - Prerequisites and error handling
   - Best for: In-depth understanding

3. **[predict_image_quick_reference.md](./predict_image_quick_reference.md)** - Quick reference card
   - Condensed information in table format
   - Quick examples
   - Common issues and solutions
   - Best for: Quick lookup while coding

## predict_image Method Overview

### Purpose
Apply trained color correction models to new images without requiring a color checker chart.

### Basic Signature
```python
def predict_image(
    self, 
    Image: Union[str, np.ndarray], 
    show: bool = False
) -> Dict[str, np.ndarray]
```

### Quick Example
```python
from ColorCorrectionPipeline.ccp import ColorCorrection

# After training models...
cc = ColorCorrection()
# ... training code ...

# Apply to new image
results = cc.predict_image(Image="test.jpg")

# Access results
final_image = results['CC']  # Final corrected image
```

## Inputs Summary

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `Image` | `str` or `np.ndarray` | Yes | Input image as file path or RGB array (float64, 0-1) |
| `show` | `bool` | No | Display plots of correction stages (default: False) |

## Outputs Summary

Returns a dictionary with 4 keys:

| Key | Description | Always Present? |
|-----|-------------|-----------------|
| `'FFC'` | Flat-field corrected image | Yes |
| `'GC'` | Gamma corrected image | Yes |
| `'WB'` | White-balanced image | Yes |
| `'CC'` | Final color-corrected image | Yes (but may be `None`) |

All output arrays:
- Shape: `(height, width, 3)`
- dtype: `float64`
- Range: `[0, 1]`
- Color space: RGB

## Processing Pipeline

```
Input → FFC → GC → WB → CC → Output Dictionary
```

Each stage applies its correction if a model exists, otherwise passes the image unchanged.

## Prerequisites

Before using `predict_image`:
1. Create `ColorCorrection` instance
2. Train models using `run()` OR load saved models
3. (Optional) Set reference illuminant

## See Also

- **Main Package README**: `../README.md` - Package installation and overview
- **Test Suite**: `../tests/test_predict_image.py` - Validation tests
- **Source Code**: `../ColorCorrectionPipeline/ccp.py` - Implementation

## Contributing

To improve this documentation:
1. Check accuracy against source code in `ColorCorrectionPipeline/ccp.py`
2. Add examples that demonstrate real-world usage
3. Update Q&A sections based on common user questions
4. Ensure examples are tested and work correctly

## Version

This documentation corresponds to:
- **Package Version**: 1.2.01+
- **Source File**: `ColorCorrectionPipeline/ccp.py`
- **Method**: `ColorCorrection.predict_image` (lines 622-704)
