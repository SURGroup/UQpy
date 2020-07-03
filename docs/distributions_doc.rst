.. _distributions_doc:

Distributions
=============

.. automodule:: UQpy.Distributions

Note that the various classes of the ``Distributions`` module are written to be consistent with distribuitons in the
``scipy.stats`` package [1]_, to the extent possible while maintaining an extensible, object oriented architecture that is
convenient for operating with the other ``UQpy`` modules. All existing distributions and their methods in ``UQpy`` are restructured from the ``scipy.stats`` package. 


Parent Distribution Class
----------------------------------

.. autoclass:: UQpy.Distributions.Distribution
	:members:
	
	
1D Continuous Distributions
---------------------------------------

In ``UQpy``, univariate continuous distributions inherit from the ``DistributionContinuous1D`` class:

.. autoclass:: UQpy.Distributions.DistributionContinuous1D
    :members: 
	
List of 1D Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
The following is a list of all 1D continuous distributions currently available in ``UQpy``.

.. autoclass:: UQpy.Distributions.Beta

.. autoclass:: UQpy.Distributions.Cauchy

.. autoclass:: UQpy.Distributions.ChiSquare

.. autoclass:: UQpy.Distributions.Exponential

.. autoclass:: UQpy.Distributions.Gamma

.. autoclass:: UQpy.Distributions.GenExtreme

.. autoclass:: UQpy.Distributions.InvGauss

.. autoclass:: UQpy.Distributions.Laplace

.. autoclass:: UQpy.Distributions.Levy

.. autoclass:: UQpy.Distributions.Logistic

.. autoclass:: UQpy.Distributions.Lognormal

.. autoclass:: UQpy.Distributions.Maxwell

.. autoclass:: UQpy.Distributions.Normal

.. autoclass:: UQpy.Distributions.Pareto

.. autoclass:: UQpy.Distributions.Rayleigh

.. autoclass:: UQpy.Distributions.TruncNorm

.. autoclass:: UQpy.Distributions.Uniform

1D Discrete Distributions
----------------------------------

In ``UQpy``, univariate discrete distributions inherit from the ``DistributionDiscrete1D`` class:

.. autoclass:: UQpy.Distributions.DistributionDiscrete1D
    :members: 
	
List of 1D Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
The following is a list of all 1D continuous distributions currently available in ``UQpy``.

.. autoclass:: UQpy.Distributions.Binomial

.. autoclass:: UQpy.Distributions.Poisson
	
Multivariate Distributions
----------------------------------

In ``UQpy``, multivariate distributions inherit from the ``DistributionND`` class:

.. autoclass:: UQpy.Distributions.DistributionND


``UQpy`` has some inbuilt multivariate distributions, which are directly child classes of ``DistributionND``. Additionally, joint distributions can be built from their marginals through the use of the ``JointInd`` and ``JointCopula`` classes described below.

List of Multivariate Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.Multinomial

.. autoclass:: UQpy.Distributions.MVNormal
    
Joint from independent marginals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.JointInd

Joint from marginals and copula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.JointCopula	


Copula
-----------

.. autoclass:: UQpy.Distributions.Copula
	:members:
	


List of Copulas
~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.Gumbel

.. autoclass:: UQpy.Distributions.Clayton

.. autoclass:: UQpy.Distributions.Frank


User-defined Distributions and Copulas
---------------------------------------------------

Defining custom distributions in ``UQpy`` can be done by sub-classing the appropriate parent class. The subclasses must possess the desired methods, per the parent ``Distribution`` class. 

Custom copulas can be similarly defined by subclassing the ``Copula`` class and defining the appropriate methods.

.. toctree::
    :maxdepth: 2

.. [1] https://docs.scipy.org/doc/scipy/reference/stats.html

	
	