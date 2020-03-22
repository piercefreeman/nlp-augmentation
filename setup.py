#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup


with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    # NOTE: @pierce mar-14-2020: we temporarily pin the versions of dependencies to the
    # same versions that were used in the original Google paper.  This allows for cross-
    # compatibility of their weight files.    
    "numpy==1.16.2",
    "tensor2tensor>=1.15.4",
    "nltk>=3.4.5",
    "requests>=2.23.0",
    "tqdm>=4.43.0",
]

setup_requirements = []

test_requirements = [
    "flake8>=3.7.9",
    "flake8-isort>=2.8.0",
    "isort>=4.3.21",
]

setup(
    author="Pierce Freeman",
    author_email="pierce.freeman@globality.com",
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Utilities for NLP Data augmentation.",
    entry_points={
        "console_scripts": [
            "uda=nlp_augmentation.uda:uda",
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords="nlp_augmentation",
    name="nlp_augmentation",
    packages=find_packages(include=["nlp_augmentation", "nlp_augmentation.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        "test": test_requirements,
        "cpu": [
            # Mirror versions required by tensor2tensor
            "tensorflow>=1.15.0,<2.0",
        ],
        "gpu": [
            "tensorflow-gpu>=1.15.0,<2.0",
        ]
    },
    url="https://github.com/piercefreeman/nlp_augmentation",
    version="0.1.0",
    zip_safe=False,
)
