#!/usr/bin/env python3
# Copyright (c) 2022-2025, GapONet Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Installation script for the 'gaponet' package."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaponet",
    version="0.1.0",
    description="GapONet: Sim-to-Real Humanoid Robot Control with DeepONet, Transformer, and MLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GapONet Developers",
    author_email="",
    url="https://github.com/yourusername/gaponet",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    python_requires=">=3.10",
    install_requires=[
        "isaaclab",
        "isaaclab-assets",
        "isaaclab-mimic",
        "isaaclab-rl",
        "isaaclab-tasks",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "pinocchio>=2.6.0",
        "pytorch-kinematics>=0.0.1",
        "psutil>=5.9.0",
        "toml>=0.10.2",
    ],
    extras_require={
        "dev": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "matplotlib>=3.5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)

