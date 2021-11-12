:github_url: https://github.com/SURGroup/UQpy

Welcome to UQpy's documentation!
================================

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox 
for modeling uncertainty in physical and mathematical systems. The code is organized 
as a set of modules centered around core capabilities in Uncertainty Quantification (UQ).

Introduction
==============

Dependencies required::

    macOS, Linux, Windows
    Python >= 3.6


Installation
-------------

Installation on a macOS can be made with the following commands.

Using Python package index (PyPI)::

    pip install UQpy

Using Conda::

    conda install --channel  ``SURG_JHU``  uqpy
    conda install -c conda-forge uqpy


From GitHub: Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version::

    git clone https://github.com/SURGroup/UQpy.git
    cd UQpy
    python setup.py {version} install


Development
-----------

UQpy is designed to serve as a platform for developing new UQ methodologies and algorithms. To install ``UQpy`` as a developer run::

    python setup.py develop

Referencing
-------------

If you are using this software in a work that will be published, please cite this paper:

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general purpose Python package and development environment for uncertainty quantification". Journal of Computational Science.

Help & Support
---------------------------

For assistance with the ``UQpy`` software package, please raise an issue on the Github `issues`_ page. Please use the appropriate labels to indicate which module you are specifically inquiring about. Alternatively, please contact ``uqpy.info@gmail.com``.


.. _issues: https://github.com/SURGroup/UQpy/issues


.. _toc:

Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   runmodel_doc
   /dimension_reduction/index
   /distributions/index
   /inference/index
   /reliability/index
   /sampling/index
   /sensitivity/index
   /stochastic_process/index
   /surrogates/index
   /transformations/index
   utilities_doc
   news_doc
   bibliography.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`