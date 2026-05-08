#!/usr/bin/env python
"""
Pre-publication verification script for ColorCorrectionPackage v1.4.5
Checks all critical files, configurations, and runtime dependencies before GitHub push.
"""

import os
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"{GREEN}✓{RESET} {description}: {filepath}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {filepath} - NOT FOUND")
        return False

def check_version_consistency():
    """Check version consistency across files"""
    print(f"\n{BLUE}Checking version consistency...{RESET}")
    
    versions = {}
    
    # Check pyproject.toml
    pyproject_path = "pyproject.toml"
    if Path(pyproject_path).exists():
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('version ='):
                    versions['pyproject.toml'] = line.split('=')[1].strip().strip('"')
                    break
    
    # Check __version__.py
    version_file = "ColorCorrectionPipeline/__version__.py"
    if Path(version_file).exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('__version__'):
                    versions['__version__.py'] = line.split('=')[1].strip().strip('"').strip("'")
                    break
    
    # Check CHANGELOG.md
    changelog_path = "CHANGELOG.md"
    if Path(changelog_path).exists():
        with open(changelog_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('## [1.4'):
                    # Extract version from header like '## [1.4.2]'
                    ver = line.strip().split('[')[1].split(']')[0]
                    versions['CHANGELOG.md'] = ver
                    break
    
    # Verify all versions match
    unique = set(versions.values())
    if len(unique) == 1:
        ver = unique.pop()
        print(f"{GREEN}\u2713{RESET} All versions consistent: {ver}")
        return True
    else:
        print(f"{RED}✗{RESET} Version mismatch:")
        for file, version in versions.items():
            print(f"  {file}: {version}")
        return False

def check_package_structure():
    """Check package structure"""
    print(f"\n{BLUE}Checking package structure...{RESET}")
    
    required_dirs = [
        "ColorCorrectionPipeline",
        "ColorCorrectionPipeline/core",
        "ColorCorrectionPipeline/flat_field",
        "ColorCorrectionPipeline/flat_field/models",
        "ColorCorrectionPipeline/io",
        "ReadMe_Images",
        ".github",
        ".github/workflows",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"{GREEN}✓{RESET} Directory exists: {dir_path}")
        else:
            print(f"{RED}✗{RESET} Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def check_required_files():
    """Check required files"""
    print(f"\n{BLUE}Checking required files...{RESET}")
    
    required_files = {
        "pyproject.toml": "Project configuration",
        "README.md": "README",
        ".gitignore": "Git ignore rules",
        "LICENSE": "License file",
        "ColorCorrectionPipeline/__init__.py": "Package init",
        "ColorCorrectionPipeline/__version__.py": "Version file",
        "ColorCorrectionPipeline/pipeline.py": "Main pipeline",
        "ColorCorrectionPipeline/config.py": "Configuration",
        "ColorCorrectionPipeline/models.py": "Models",
        "ColorCorrectionPipeline/constants.py": "Constants",
        "ColorCorrectionPipeline/flat_field/models/plane_det_model_YOLO_512_n.onnx": "OpenCV DNN plane detection model",
        ".github/workflows/publish-to-pypi.yml": "GitHub Actions workflow",
    }
    
    all_exist = True
    for filepath, description in required_files.items():
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def check_pyproject_toml():
    """Check pyproject.toml configuration"""
    print(f"\n{BLUE}Checking pyproject.toml configuration...{RESET}")
    
    if not Path("pyproject.toml").exists():
        print(f"{RED}✗{RESET} pyproject.toml not found")
        return False
    
    with open("pyproject.toml", 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'packages = ["ColorCorrectionPipeline"]': "Correct package name",
        'scikit-image>=0.19.0': "scikit-image dependency",
        'requires-python = ">=3.8"': "Python version requirement",
    }
    
    all_ok = True
    for check, description in checks.items():
        if check in content:
            print(f"{GREEN}✓{RESET} {description}")
        else:
            print(f"{RED}✗{RESET} {description} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_readme():
    """Check README.md content"""
    print(f"\n{BLUE}Checking README.md content...{RESET}")
    
    if not Path("README.md").exists():
        print(f"{RED}✗{RESET} README.md not found")
        return False
    
    with open("README.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'ColorCorrectionPipeline': "Package name",
        '## Package Structure': "Package structure section",
        'from ColorCorrectionPipeline import ColorCorrection, Config': "Correct import",
        'from ColorCorrectionPipeline.core.utils import to_float64': "Core utils import",
        '## Sample Results': "Sample results section",
        'ReadMe_Images/before.svg': "Before image",
        'ReadMe_Images/After.svg': "After image",
    }
    
    all_ok = True
    for check, description in checks.items():
        if check in content:
            print(f"{GREEN}✓{RESET} {description}")
        else:
            print(f"{YELLOW}⚠{RESET} {description} - CHECK MANUALLY")
            all_ok = False
    
    return all_ok

def check_github_workflow():
    """Check GitHub Actions workflow"""
    print(f"\n{BLUE}Checking GitHub Actions workflow...{RESET}")
    
    workflow_path = ".github/workflows/publish-to-pypi.yml"
    if not Path(workflow_path).exists():
        print(f"{RED}✗{RESET} Workflow file not found")
        return False
    
    with open(workflow_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'name: Publish to PyPI': "Workflow name",
        'pyproject.toml': "Trigger on pyproject.toml",
        '__version__.py': "Trigger on version file",
        'PYPI_CCP_TOKEN': "PyPI token reference",
        'twine upload': "PyPI upload step",
        'git tag': "Tag creation",
    }
    
    all_ok = True
    for check, description in checks.items():
        if check in content:
            print(f"{GREEN}✓{RESET} {description}")
        else:
            print(f"{RED}✗{RESET} {description} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_critical_modules():
    """Verify that critical runtime modules are available."""
    print(f"\n{BLUE}Checking critical runtime modules...{RESET}")

    all_ok = True

    # cv2 + cv2.mcc
    try:
        import cv2
        print(f"{GREEN}✓{RESET} cv2 {cv2.__version__}")
        if hasattr(cv2, "mcc"):
            print(f"{GREEN}✓{RESET} cv2.mcc module present")
        else:
            print(f"{RED}✗{RESET} cv2.mcc module MISSING — install opencv-contrib-python")
            all_ok = False
    except ImportError:
        print(f"{RED}✗{RESET} cv2 not importable")
        all_ok = False

    # numba
    try:
        import numba
        print(f"{GREEN}✓{RESET} numba {numba.__version__}")
    except ImportError:
        print(f"{RED}✗{RESET} numba not importable")
        all_ok = False

    # ColorCorrectionPipeline self-import
    try:
        from ColorCorrectionPipeline import __version__
        from ColorCorrectionPipeline.core import HAS_MCC
        print(f"{GREEN}✓{RESET} ColorCorrectionPipeline {__version__}")
        if HAS_MCC:
            print(f"{GREEN}✓{RESET} HAS_MCC = True")
        else:
            print(f"{RED}✗{RESET} HAS_MCC = False — color correction will fail")
            all_ok = False
    except ImportError as e:
        print(f"{RED}✗{RESET} ColorCorrectionPipeline import failed: {e}")
        all_ok = False

    # numpy, pandas, scipy, sklearn
    for mod_name in ["numpy", "pandas", "scipy", "sklearn"]:
        try:
            __import__(mod_name)
            print(f"{GREEN}✓{RESET} {mod_name}")
        except ImportError:
            print(f"{RED}✗{RESET} {mod_name} not importable")
            all_ok = False

    return all_ok

def main():
    """Run all verification checks"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}ColorCorrectionPackage v1.4.5 Pre-Publication Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    results = []
    
    # Run all checks
    results.append(("Version Consistency", check_version_consistency()))
    results.append(("Package Structure", check_package_structure()))
    results.append(("Required Files", check_required_files()))
    results.append(("pyproject.toml", check_pyproject_toml()))
    results.append(("README.md", check_readme()))
    results.append(("GitHub Workflow", check_github_workflow()))
    results.append(("Critical Modules", check_critical_modules()))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Verification Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    all_passed = True
    for check_name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_passed:
        print(f"{GREEN}✓ All checks passed! Ready for GitHub publication.{RESET}")
        print(f"\n{YELLOW}Next steps:{RESET}")
        print(f"1. Set up PYPI_CCP_TOKEN secret in GitHub")
        print(f"2. Run: git add . && git commit -m 'Release v1.4.5'")
        print(f"3. Run: git push origin main")
        print(f"4. Monitor GitHub Actions workflow")
        print(f"5. Create GitHub release after successful PyPI upload")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Please fix issues before publishing.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
