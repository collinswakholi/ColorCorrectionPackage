# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-09-25

### Added
- Modern `pyproject.toml` configuration for improved packaging
- Automatic inclusion of YOLO model (`plane_det_model_YOLO_512_n.pt`) in package distribution
- Comprehensive README with enhanced installation instructions
- Development dependencies and tooling configuration
- Contributing guidelines and development setup instructions
- Proper package metadata and classifiers for PyPI
- Citation information and enhanced documentation

### Changed
- Upgraded package structure for better GitHub publishing
- Enhanced setup.py with detailed project metadata
- Improved MANIFEST.in for better file inclusion control
- Updated README with modern installation methods
- Enhanced package documentation and usage examples

### Fixed
- YOLO model file now automatically included in package installations
- Improved package data handling and distribution
- Better dependency management and requirements specification

## [1.1.x] - Previous Versions

### Added
- Core color correction pipeline functionality
- Flat-field correction (FFC) with manual and automatic modes
- Gamma correction (GC) with polynomial fitting
- White balance (WB) correction
- Color correction (CC) with conventional and custom methods
- Support for various image processing tasks
- Integration with scientific libraries (colour-science, scikit-learn, etc.)

### Features
- Multi-stage color correction pipeline
- Support for color checker based corrections
- Automatic and manual ROI selection
- Multiple correction algorithms (linear, PLS, neural networks)
- Comprehensive metrics and validation
- Prediction on new images without color charts

---

For more detailed changes, see the [commit history](https://github.com/collinswakholi/ColorCorrectionPackage/commits/main).