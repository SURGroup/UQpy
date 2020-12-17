#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='UQpy',
    version="3.0.2",
    url='https://github.com/SURGroup/UQpy',
    description="UQpy is a general purpose toolbox for Uncertainty Quantification",
    author="Michael D. Shields, Dimitris G. Giovanis, Audrey Olivier, Aakash Bangalore-Satish, Mohit Chauhan, "
           "Lohit Vandanapu, Ketson R.M. dos Santos",
    author_email="UQpy.info@gmail.com",
    license='MIT',
    platforms=["OSX", "Windows", "Linux"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.pdf"]},
    install_requires=[
        "numpy", "scipy", "matplotlib", "scikit-learn", 'fire'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
