:github_url: https://github.com/SURGroup/UQpy

Welcome to UQpy's documentation!
================================

UQpy (Uncertainty Quantification with python) is a general purpose python toolbox
for modeling uncertainty in physical and mathematical systems. The code is organized 
as a set of modules centered around core capabilities in Uncertainty Quantification (UQ).

------------

+-----------------------+------------------------------------------------------------------+
| **Product Owner:**    | Michael D. Shields                                               |
+-----------------------+------------------------------------------------------------------+
| **Lead Developers:**  | Dimitris Giovanis, Audrey Olivier, Dimitris Tsapetis             |
+-----------------------+------------------------------------------------------------------+
| **Development Team:** | Aakash Bangalore Satish, Mohit Singh Chauhan, Lohit Vandanapu,   |
+                       +                                                                  +
|                       | Ketson RM dos Santos, Katiana Kontolati, Dimitris Loukrezis,     |
+                       +                                                                  +
|                       | Promit Chakroborty, Lukáš Novák, Andrew Solanto, Connor Krill    |
+-----------------------+------------------------------------------------------------------+
| **Contributors:**     | Michael Gardner, Prateek Bhustali, Julius Schultz, Ulrich Römer  |
+-----------------------+------------------------------------------------------------------+

Introduction
------------

Dependencies required::

    macOS, Linux, or Windows
    Python >= 3.9


------------

Installation
-------------

:py:mod:`UQpy` can be installed with the following commands.

Using Python package index (PyPI)::

MacOS or Linux::

    pip install UQpy


Windows::

    pip3 install UQpy


Using Conda::

    conda install --channel  ``SURG_JHU``  uqpy
    conda install -c conda-forge uqpy


From GitHub: Clone your fork of the :py:mod:`UQpy` repo from your GitHub account to your local disk (to get the latest version)::

    git clone https://github.com/SURGroup/UQpy.git
    cd UQpy
    python setup.py {version} install

------------

Development
-----------

:py:mod:`UQpy` is designed to serve as a platform for developing new UQ methodologies and algorithms. To install :py:mod:`UQpy` as a developer, run::

    python setup.py {version} develop


------------

Logging
-----------

:py:mod:`UQpy` adopts the built-in :py:mod:`logging` python library. This allows for a fine-grain logging of events of various severity levels.
The available logging levels allowed by the library are:

**DEBUG**, **INFO**, **WARNING**, **ERROR**, **CRITICAL**

The default logging level is set to **ERROR**. The user can change the logging severity level by including the following line in their code and choosing the desired logging level:

.. code-block:: python
  :linenos:

  logging.getLogger('UQpy').setLevel(logging.INFO)


.. toctree::
   :hidden:

   Home <self>
   /dimension_reduction/index
   /distributions/index
   /inference/index
   /reliability/index
   runmodel_doc
   /sampling/index
   /sensitivity/index
   /stochastic_process/index
   /surrogates/index
   /transformations/index
   /utilities/index
   architecture.rst
   paper.rst
   bibliography.rst
   news_doc


------------

Referencing UQpy
----------------

If you are using this software in a work that will be published, please cite this paper:

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general purpose Python package and development environment for uncertainty quantification". Journal of Computational Science. Volume 47, 101204

https://doi.org/10.1016/j.jocs.2020.101204

Examples from the above article were performed using :py:mod:`UQpy` version 3. These examples can be found at:

https://github.com/SURGroup/UQpy_paper

This repository contains a binder link to execute portions of the code.

------------

Contact
------------

To engage in conversations about uncertainty quantification, or ask questions about :py:mod:`UQpy` usage and functionality refer to the :py:mod:`UQpy` discussions tab:

`Discussions <https://github.com/SURGroup/UQpy/discussions>`_

Help & Support
---------------------------

For assistance with the :py:mod:`UQpy` software package, please raise an issue on the Github `issues`_ page. Please use the appropriate labels to indicate which module you are specifically inquiring about.


.. _issues: https://github.com/SURGroup/UQpy/issues



