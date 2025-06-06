import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="ColorCorrectionPipeline",
    version="1.1.2",
    author="Collins Wakholi, Devin A. Rippner",
    author_email="wcoln@yahoo.com, devinrippner@gmail.com",
    description="A Stepwise color‐correction pipeline with flat‐field, gamma, white‐balance, and color‐correction stages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/collinswakholi/ColorCorrectionPackage",
    project_urls={
        "Bug Tracker": "https://github.com/collinswakholi/ColorCorrectionPackage/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["color", "image-processing", "flat-field", "gamma-correction", "white-balance", "color-correction"],
    package_dir={"": "src"},  # Tells setuptools your packages are under 'src/'
    packages=setuptools.find_packages(where="src"), # Automatically finds packages in 'src/'
    python_requires=">=3.8", # You should define your minimum Python version here if not already
    install_requires=requirements,
    include_package_data=True
)