from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-quantization",
    version="1.0.0",
    author="Yash Darji",
    author_email="yash@example.com",
    description="Production-grade neural network quantization achieving <2% degradation, 4× compression, 3.2× speedup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yash2378/neural-quantization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "safetensors>=0.4.0",
        "auto-gptq>=0.7.0",
        "optimum>=1.15.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "marlin": ["marlin-cuda>=0.1.0"],
        "triton": ["triton>=2.0.0"],
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "neural-quantize=src.tools.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)