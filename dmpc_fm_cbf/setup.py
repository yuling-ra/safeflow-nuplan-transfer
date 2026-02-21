"""
Setup script for DMPC + Flow Matching + CBF package.
"""

from setuptools import setup, find_packages

setup(
    name="dmpc_fm_cbf",
    version="0.1.0",
    author="Yuling",
    description="DMPC + Flow Matching + CBF Framework for Multi-Agent Trajectory Planning",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "full": [
            "torchdiffeq>=0.2.0",
            "cvxpy>=1.2.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
