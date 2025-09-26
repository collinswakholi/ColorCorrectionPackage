# Contributing to ColorCorrectionPipeline

Thank you for your interest in contributing to ColorCorrectionPipeline! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of color correction and image processing concepts

### Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ColorCorrectionPackage.git
   cd ColorCorrectionPackage
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Installing Development Dependencies

The development dependencies include testing, linting, and formatting tools:

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pre-commit` - Git hooks

### Pre-commit Hooks (Optional but Recommended)

Set up pre-commit hooks to automatically format and check your code:

```bash
pre-commit install
```

## Making Changes

### Branching Strategy

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Use descriptive branch names**:
   - `feature/add-new-correction-method`
   - `fix/gamma-correction-bug`
   - `docs/improve-readme`

### Code Changes

1. **Make your changes** in logical, atomic commits
2. **Follow the existing code style** and patterns
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Test your changes** thoroughly

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for custom color spaces

- Implement RGB to Lab color space conversion
- Add unit tests for color space transformations
- Update documentation with new color space options

Fixes #123
```

## Testing

### Running Tests

Run the full test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=ColorCorrectionPipeline
```

Run specific test files:
```bash
pytest tests/test_specific_module.py
```

### Writing Tests

- Place test files in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Include both unit tests and integration tests
- Test edge cases and error conditions

Example test structure:
```python
import pytest
from ColorCorrectionPipeline.ccp import ColorCorrection

def test_color_correction_initialization():
    """Test that ColorCorrection can be initialized properly."""
    cc = ColorCorrection()
    assert cc is not None

def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    cc = ColorCorrection()
    with pytest.raises(ValueError):
        cc.process_invalid_input()
```

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run tests and checks**:
   ```bash
   pytest
   black .
   flake8 .
   ```

3. **Push your branch**:
   ```bash
   git push origin your-feature-branch
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots if applicable
   - Notes about testing performed

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or breaking changes documented)
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Clear title** and description
2. **Steps to reproduce** the issue
3. **Expected behavior**
4. **Actual behavior**
5. **Environment information**:
   - Python version
   - Operating system
   - Package version
   - Relevant dependencies

### Feature Requests

When requesting features:

1. **Describe the problem** you're trying to solve
2. **Explain the proposed solution**
3. **Provide use cases** and examples
4. **Consider implementation complexity**

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import organization
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for public functions

### Code Formatting

Use `black` for automatic code formatting:
```bash
black .
```

### Documentation

- Update docstrings for any modified functions
- Use clear, concise language
- Include examples where helpful
- Update README if user-facing changes are made

### Example Function Documentation

```python
def process_image(image: np.ndarray, correction_type: str = "gamma") -> np.ndarray:
    """Process an image with the specified correction type.
    
    Args:
        image: Input image as numpy array in RGB format, values [0, 1]
        correction_type: Type of correction to apply ("gamma", "white_balance", "color")
        
    Returns:
        Corrected image as numpy array in RGB format, values [0, 1]
        
    Raises:
        ValueError: If correction_type is not supported
        
    Examples:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)
        >>> corrected = process_image(img, "gamma")
    """
```

## Questions?

If you have questions about contributing:

1. Check the [documentation](README.md)
2. Search [existing issues](https://github.com/collinswakholi/ColorCorrectionPackage/issues)
3. Open a new issue with the "question" label
4. Contact the maintainers

## Recognition

Contributors will be acknowledged in:
- README.md contributors section  
- CHANGELOG.md for significant contributions
- Release notes

Thank you for contributing to ColorCorrectionPipeline! 🎨