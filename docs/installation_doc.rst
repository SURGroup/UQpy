.. _installation_doc:

Installation
============

Dependencies required::
    
	macOS, Linux, Windows
	Python >= 3.6
	Git >= 2.13.1

Installation on a macOS can be made with the following commands.

Using Python package index (PyPI)::

	pip install UQpy
	
Using Conda::

	conda install --channel  ``SURG_JHU``  uqpy
	

From GitHub: Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version::

	git clone https://github.com/SURGroup/UQpy.git
	cd UQpy
	python setup.py install 


Development
-----------

UQpy is designed to serve as a platform for developing new UQ methodologies and algorithms. To install ``UQpy`` as a developer run::

    python setup.py develop 

Documentation
-------------

If you are using this software in a work that will be published, please cite this
paper as ...

Help & Support
---------------------------

For any problems and questions you might have related to ``UQpy``, please contact ``uqpy.info@gmail.com``.
