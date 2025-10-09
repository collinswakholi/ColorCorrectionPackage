# Tests Directory

This directory contains test suites for the ColorCorrectionPipeline package.

## Test Files

### test_basic.py
Basic tests for package imports and initialization.

### test_predict_image.py
Comprehensive test suite for the `predict_image` method in the `ColorCorrection` class.

## Running Tests

### Run all tests with pytest
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_predict_image.py -v
```

### Run tests directly (without pytest)
```bash
python3 tests/test_predict_image.py
```

## Test Coverage for predict_image

The `test_predict_image.py` file includes 23 comprehensive tests organized into 7 test classes:

### 1. TestPredictImageBasicFunctionality
- ✓ Test with JPG file path input
- ✓ Test with PNG file path input
- ✓ Test with numpy array input
- ✓ Test output dictionary structure

### 2. TestPredictImageModelCombinations
- ✓ Test with no models (pass-through)
- ✓ Test with only FFC model
- ✓ Test with only GC model
- ✓ Test with only WB model
- ✓ Test with only CC model (conventional method)
- ✓ Test with only CC model (custom/ours method)

### 3. TestPredictImageEdgeCases
- ✓ Test invalid file path (raises FileNotFoundError)
- ✓ Test invalid input type (raises TypeError)
- ✓ Test different image sizes
- ✓ Test grayscale image handling

### 4. TestPredictImageValidation
- ✓ Test output shapes match input
- ✓ Test pixel value ranges (0.0 to 1.0)
- ✓ Test None values when models are missing

### 5. TestPredictImagePerformance
- ✓ Test execution time tracking
- ✓ Test multiple predictions consistency

### 6. TestPredictImageVisualOutput
- ✓ Test show=False parameter
- ✓ Test show=True parameter
- ✓ Test saving output images

### 7. TestPredictImageIntegration
- ✓ Test with real image data from Data/Images/

## Test Results

All 23 tests pass successfully:
```
============================= test session starts ==============================
tests/test_predict_image.py .......................                      [100%]
============================== 23 passed in 3.27s ==============================
```

## Requirements

The tests require the following dependencies (already in requirements.txt):
- pytest
- numpy
- opencv-python
- colour-science
- All ColorCorrectionPipeline dependencies

## Test Output

The tests create temporary files in `/tmp/test_predict_image/` and `/tmp/test_predict_image_output/` for:
- Synthetic test images (JPG, PNG)
- Output images from each processing stage (FFC, GC, WB, CC)

These temporary files are automatically cleaned up by the OS.

## Test Features

- **Comprehensive coverage**: Tests all aspects of the predict_image method
- **Independent tests**: Each test is isolated and can run independently
- **Detailed logging**: INFO level logging for debugging
- **Mock models**: Uses synthetic models to test each stage
- **Real data integration**: Uses actual images from Data/Images/ when available
- **Performance tracking**: Measures and validates execution time
- **Visual validation**: Saves output images for manual inspection
- **Error handling**: Tests both success and failure scenarios

## Adding New Tests

To add new tests to `test_predict_image.py`:

1. Create a new test class inheriting from `unittest.TestCase`
2. Add a `setUpClass` method for shared fixtures
3. Add test methods starting with `test_`
4. Include docstrings explaining what each test does
5. Use logging to provide detailed information
6. Add the test class to the `run_tests_with_summary()` function

Example:
```python
class TestNewFeature(unittest.TestCase):
    """Test description."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.cc = ColorCorrection()
    
    def test_new_feature(self):
        """Test specific feature."""
        logger.info("Testing new feature...")
        result = self.cc.predict_image(...)
        self.assertIsNotNone(result)
        logger.info("✓ New feature test passed")
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines. They:
- Don't require a display (headless compatible)
- Have reasonable execution time (< 10 seconds total)
- Use temporary directories for test artifacts
- Handle missing dependencies gracefully
- Provide clear pass/fail status
