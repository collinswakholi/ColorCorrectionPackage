import setuptools
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements and clean them
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="ColorCorrectionPipeline",
    version="1.2.0",
    author="Collins Wakholi, Devin A. Rippner",
    author_email="wcoln@yahoo.com, devinrippner@gmail.com",
    description="A stepwise color‐correction pipeline with flat‐field, gamma, white‐balance, and color‐correction stages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/collinswakholi/ColorCorrectionPackage",
    project_urls={
        "Documentation": "https://github.com/collinswakholi/ColorCorrectionPackage#readme",
        "Bug Tracker": "https://github.com/collinswakholi/ColorCorrectionPackage/issues",
        "Source Code": "https://github.com/collinswakholi/ColorCorrectionPackage",
        "Release Notes": "https://github.com/collinswakholi/ColorCorrectionPackage/releases",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["color-correction", "image-processing", "flat-field", "gamma-correction", "white-balance", "computer-vision", "photography", "scientific-imaging"],
    packages=setuptools.find_packages(exclude=["tests*", "docs*"]),
    package_data={
        'ColorCorrectionPipeline.FFC.Models': ['*.pt'],
        'ColorCorrectionPipeline.Configs': ['*'],
        'ColorCorrectionPipeline.utils': ['*']
    },
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)