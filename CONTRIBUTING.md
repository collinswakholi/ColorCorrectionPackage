# Contributing to Color Correction Package

First off, thank you for considering contributing to the Color Correction Package! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful**: Treat everyone with respect and consideration
- **Be collaborative**: Work together and help each other
- **Be inclusive**: Welcome newcomers and help them contribute
- **Be professional**: Keep discussions focused and constructive

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of color correction and image processing
- Familiarity with NumPy, OpenCV, and PyTorch

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ColorCorrectionPackage.git
   cd ColorCorrectionPackage
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Run tests** to verify setup:
   ```bash
   pytest tests/
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue list](https://github.com/collinswakholi/ColorCorrectionPackage/issues) to avoid duplicates.

When creating a bug report, include:

- **Clear title**: Descriptive summary of the issue
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, package version
- **Code sample**: Minimal code that reproduces the issue
- **Screenshots**: If applicable

**Bug Report Template:**

```markdown
**Description:**
A clear description of the bug.

**To Reproduce:**
1. Step 1
2. Step 2
3. See error

**Expected Behavior:**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- Package version: [e.g., 1.3.0]

**Code Sample:**
```python
# Minimal code to reproduce
```

**Error Message:**
```
Full error traceback
```
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear title**: Describe the enhancement
- **Provide detailed description**: Explain why this enhancement would be useful
- **Include examples**: Show how the enhancement would work
- **Consider alternatives**: Discuss other approaches you've considered

### Contributing Code

We love pull requests! Here's how to contribute code:

1. **Find an issue** to work on or create a new one
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** with clear, focused commits
5. **Add tests** for your changes
6. **Run the test suite** to ensure nothing breaks
7. **Update documentation** if needed
8. **Submit a pull request**

## Development Setup

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/collinswakholi/ColorCorrectionPackage.git
cd ColorCorrectionPackage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install
```

### Project Structure

```
color_correc_optim/
├── color_correc_optim/          # Main package
│   ├── core/                    # Core algorithms
│   ├── flat_field/              # FFC module
│   ├── io/                      # I/O utilities
│   ├── pipeline.py              # Main pipeline
│   ├── models.py                # Model management
│   └── config.py                # Configuration
├── tests/                       # Test suite
│   └── test_unit/               # Unit tests
├── scripts/                     # Development scripts
├── examples/                    # Example configs
└── docs/                        # Documentation
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_unit/test_color_spaces.py

# Run with coverage
pytest --cov=color_correc_optim tests/

# Run equivalence tests
python scripts/test_pipeline_equivalence.py

# Run benchmarks
python scripts/benchmark_pipeline.py
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized (standard library, third-party, local)

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all files
black color_correc_optim/

# Check formatting without changes
black --check color_correc_optim/
```

### Type Hints

Use type hints for all public APIs:

```python
from typing import Optional, Union, Dict, Tuple
import numpy as np

def process_image(
    image: Union[str, np.ndarray],
    config: Optional[Config] = None
) -> Tuple[Dict, bool]:
    """Process an image with optional configuration."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ColorCorrection`, `MyModels`)
- **Functions/methods**: `snake_case` (e.g., `predict_image`, `apply_ffc`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_PATH`, `DEFAULT_BINS`)
- **Private members**: `_leading_underscore` (e.g., `_internal_method`)

## Testing Guidelines

### Writing Tests

- **Test one thing**: Each test should verify one specific behavior
- **Use descriptive names**: Test names should clearly indicate what they test
- **Follow AAA pattern**: Arrange, Act, Assert
- **Use fixtures**: For common setup code

Example test:

```python
import pytest
import numpy as np
from color_correc_optim import ColorCorrection, Config

def test_color_correction_basic_pipeline():
    """Test that ColorCorrection runs basic pipeline without errors."""
    # Arrange
    cc = ColorCorrection()
    img = np.random.rand(480, 640, 3).astype(np.float64)
    white = (np.ones((480, 640, 3)) * 255).astype(np.uint8)
    config = Config(do_ffc=False, do_gc=False, do_wb=False, do_cc=True)
    
    # Act
    metrics, images, error = cc.run(Image=img, White_Image=white, config=config)
    
    # Assert
    assert not error, "Pipeline should complete without errors"
    assert 'final' in images, "Should return final image"
    assert images['final'].shape == img.shape, "Output shape should match input"
```

### Test Coverage

- Aim for **>80% code coverage**
- Focus on **critical paths** and **edge cases**
- Test **error handling** thoroughly
- Include **integration tests** for key workflows

### Running Specific Tests

```bash
# Run tests matching pattern
pytest -k "test_color"

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Pull Request Process

### Before Submitting

1. ✅ **Run all tests** and ensure they pass
2. ✅ **Check code formatting** with Black
3. ✅ **Update documentation** if needed
4. ✅ **Add/update tests** for your changes
5. ✅ **Update CHANGELOG.md** with your changes
6. ✅ **Ensure no merge conflicts** with main branch

### PR Guidelines

- **One PR per feature**: Keep changes focused
- **Clear title**: Summarize the change
- **Detailed description**: Explain what, why, and how
- **Reference issues**: Link to related issues
- **Screenshots**: Include if UI/visual changes
- **Breaking changes**: Clearly document any breaking changes

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Related Issues
Fixes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented if necessary)

## Screenshots (if applicable)
Add screenshots here.
```

### Review Process

1. **Automated checks**: CI/CD will run tests automatically
2. **Code review**: Maintainer will review your code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Once approved, your PR will be merged
5. **Recognition**: You'll be added to contributors list!

### Merging

- PRs require **at least one approval** from a maintainer
- All **CI checks must pass**
- **No merge conflicts** with main branch
- Maintainers will merge using **squash and merge** for clean history

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs or request features
- **Documentation**: Check the [docs/](docs/) folder

### Maintainers

- **Collins Wakholi** ([@collinswakholi](https://github.com/collinswakholi)) - Creator and Lead Maintainer

### Recognition

Contributors are recognized in:
- **README.md** contributors section
- **CHANGELOG.md** for each version
- GitHub's contributors page

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open a [discussion](https://github.com/collinswakholi/ColorCorrectionPackage/discussions)
- Create an [issue](https://github.com/collinswakholi/ColorCorrectionPackage/issues)
- Contact the maintainers

---

**Thank you for contributing to the Color Correction Package!** 🎉

Your contributions help make color correction more accessible to everyone.
