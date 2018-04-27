#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='UQpy',
    version="0.1.0",
    url='https://github.com/SURGroup/UQpy',
    authors="Dimitris G. Giovanis, Michael D. Shields",
    author_emails="dgiovan1@jhu.edu, michael.shields@jhu.edu",
    license='Apache',
    platforms='OSX',
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"":["*.pdf"]},
    install_requires=[
        "numpy", "scipy", "chaospy","pyDOE","scikit-learn"
    ],
    classifiers=[
        'Development Status :: 1 - Production/Unstable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: Apache',
        'Natural Language :: English',
    ],
)