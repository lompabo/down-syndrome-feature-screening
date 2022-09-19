#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ML4downsyndrom", # Replace with your own username
    author="Marijana Rakvin, Federico Baldo, Michele Lombardi, Allison Piovesan, Maria Chiare Pelleri",
    author_email="federico.baldo2@unibo.it",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)