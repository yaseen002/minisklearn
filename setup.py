from setuptools import setup, find_packages

setup(
    name="minisklearn",
    version="0.1.0",
    author="Your Name",
    description="A from-scratch machine learning library built with NumPy.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.8",
)