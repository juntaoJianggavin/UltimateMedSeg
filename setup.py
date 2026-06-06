from setuptools import setup, find_packages

setup(
    name="medseg",
    version="0.1.0",
    description="Modular 2D Medical Image Segmentation Framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "einops>=0.6.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
)
