
*******************************************
Uncertainty Quantification with python (UQpy)
*******************************************

|logo|

====

:Authors: Michael D. Shields, Dimitris G. Giovanis, Audrey Olivier, Aakash Bangalore Satish, Mohit Singh Chauhan, Lohit Vandanapu, Ketson RM dos Santos, Katiana Kontolati
:Contact: UQpy.info@gmail.com
:Version: 3.0.1


Description
===========

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox for modeling uncertainty in physical and mathematical systems.

Documentation
================

Website:
           https://uqpyproject.readthedocs.io

Dependencies
===========

            * ::
            
                Python >= 3.6
                Git >= 2.13.1

License
===========
UQpy is distributed under the MIT license

Copyright (C) <2018> <Michael D. Shields>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Installation
===========

From PyPI

            * ::

                        pip install UQpy

In order to uninstall it

            * ::

                        pip uninstall UQpy

Using Conda

            * ::

                        conda install --channel  ``SURG_JHU``  uqpy

Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version): 

            * ::

                        git clone https://github.com/SURGroup/UQpy.git
                        cd UQpy/
                        python setup.py install  (user installation)
                        python setup.py develop (developer installation)

Referencing UQpy
=================

If you are using this software in a work that will be published, please cite this paper:

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general purpose Python package and development environment for uncertainty quantification". Journal of Computational Science.


Help and Support
===========

For assistance with the UQpy software package, please raise an issue on the Github Issues page. Please use the appropriate labels to indicate which module you are specifically inquiring about.

.. image:: https://img.shields.io/pypi/dm/UQpy?style=plastic   :alt: PyPI - Downloads
.. image:: https://img.shields.io/conda/dn/conda-forge/UQpy?style=plastic   :alt: Conda
.. image:: https://img.shields.io/github/downloads/SURGroup/UQpy/v3.0.1/total?style=plastic   :alt: GitHub Releases (by Release)

.. image:: https://img.shields.io/pypi/v/UQpy?style=plastic   :alt: PyPI
.. image:: https://img.shields.io/conda/v/conda-forge/UQpy?style=plastic   :alt: Conda

.. |logo| image:: logo.jpg
    :scale: 25 %
    :target: https://gihub.com/SURGroup/UQpy
    
    
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/SURGroup/UQpy/master
