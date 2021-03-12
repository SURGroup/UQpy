.. _installation_doc:

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
	python setup.py install 


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
