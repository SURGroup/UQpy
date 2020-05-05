.. _distributions:

Distributions
=============

.. automodule:: UQpy.Distributions


Distribution Parent Class
-------------------------

In ``UQpy``, the ``Distribution`` class is the parent class to all probability distributions. It consists of a suite of ``scipy``-based methods for generating realizations from the distribution and estimating: the probability density function and its logarithm, the cumulative distribution function and its inverse, the moments and the maximum likelihood ``mle``.


.. autoclass:: UQpy.Distributions.Distribution
	:members:
	:private-members:

	
Sub-classing ``Distribution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Distribution`` class is used to define:
	 - Continuous Distributions
	 - Discrete Distributions
	 - Univariate (1D) Distributions
	 - Multivariate (ND) Distributions
	
Continuous Distributions
------------------------

In ``UQpy``, the ``DistributionContinuous1D`` class  is used to define a 1-dimensional continuous probability distribution and its associated methods.

.. autoclass:: UQpy.Distributions.DistributionContinuous1D
    :members: 

Discrete Distributions
----------------------

In ``UQpy``, the ``DistributionDiscrete1D`` class  is used to define a 1-dimensional discrete probability distribution and its associated methods.

.. autoclass:: UQpy.Distributions.DistributionDiscrete1D
    :members: 
	
Multivariate Distributions
--------------------------

In ``UQpy``, the ``DistributionND`` class is used to define a multivariate distribution. This can be done:
    - via direct sub-classing, see for instance the multivariate normal,
    - via a list of independent marginals, each of them being of class ``DistributionContinuous1D``,
    - via a list of marginals and a copula (of class ``Copula``) to account for dependency between dimensions.

.. autoclass:: UQpy.Distributions.DistributionND
    :members: 
	
Copula
-------

.. autoclass:: UQpy.Distributions.Copula
	:members:
	:private-members:
	
	
List of Distributions
---------------------

.. autoclass:: UQpy.Distributions.Normal

.. autoclass:: UQpy.Distributions.Genextreme

List of Copula
---------------

In ``UQpy`` the ``Copula`` class can be used to define a multivariate distribution whose dependence structure is defined with a copula. This class is used in support of the main ``Distribution`` class. The following copula are supported: 
	- Gumbel
	- Clayton
	- Frank

.. autoclass:: UQpy.Distributions.Gumbel

.. autoclass:: UQpy.Distributions.Frank

.. autoclass:: UQpy.Distributions.Clayton

.. toctree::
    :maxdepth: 2



	
	