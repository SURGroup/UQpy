:github_url: https://github.com/SURGroup/UQpy

Welcome to UQpy's documentation!
================================

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox 
for modeling uncertainty in physical and mathematical systems. The code is organized 
as a set of modules centered around core capabilities in Uncertainty Quantification (UQ).

------------

Introduction
------------

Dependencies required::

    macOS, Linux, Windows
    Python >= 3.9


------------

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

------------

Development
-----------

UQpy is designed to serve as a platform for developing new UQ methodologies and algorithms. To install ``UQpy`` as a developer run::

    python setup.py {version} develop


------------

Logging
-----------

UQpy adopts the built-in `logging` python library. This allows for a fine-grain logging of events of various severity levels.
The available logging levels allowed by the library are:

**DEBUG**, **INFO**, **WARNING**, **ERROR**, **CRITICAL**

The default logging level is set to **ERROR**. The user can change the logging severity level to allow for more detailed
listing of the occurring events. This can performed by including the following line in their code and choosing the desired logging level:

.. code-block:: python
  :linenos:

  logging.getLogger('UQpy').setLevel(logging.INFO)


.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Computational Model Evaluation

   runmodel_doc

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Basic Probabilistic Operations

   /distributions/index
   /sampling/index
   /transformations/index
   /stochastic_process/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Analysis

   /reliability/index
   /inference/index
   /surrogates/index
   /sensitivity/index
   /dimension_reduction/index


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Information

   paper.rst
   bibliography.rst
   news_doc


------------

Referencing UQpy
----------------

If you are using this software in a work that will be published, please cite this paper:

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general purpose Python package and development environment for uncertainty quantification". Journal of Computational Science. Volume 47, 101204

https://doi.org/10.1016/j.jocs.2020.101204

Examples from the above article were performed using UQpy version 3. These examples can be found at:

https://github.com/SURGroup/UQpy_paper

------------

Help & Support
---------------------------

For assistance with the ``UQpy`` software package, please raise an issue on the Github `issues`_ page. Please use the appropriate labels to indicate which module you are specifically inquiring about.


.. _issues: https://github.com/SURGroup/UQpy/issues
