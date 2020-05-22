.. _distributions:

Distributions
=============

.. automodule:: UQpy.Distributions

Parent Distribution Class
----------------------------------

.. autoclass:: UQpy.Distributions.Distribution
	:members:
	
	
1D Continuous Distributions
---------------------------------------

In ``UQpy``, univariate continuous distributions inherit from the ``DistributionContinuous1D`` class:

.. autoclass:: UQpy.Distributions.DistributionContinuous1D
    :members: 
	
List of Distributions
~~~~~~~~~~~~~~~~~~~~~~~~
	
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
	
Multivariate Distributions
----------------------------------

In ``UQpy``, multivariate distributions inherit from the ``DistributionND`` class:

.. autoclass:: UQpy.Distributions.DistributionND

Furthermore, joint distributions can be built from their marginals through the use of the ``JointInd`` and ``JointCopula`` classes described below.
    
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
	

List of Distributions
--------------------------



Univariate discrete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.Binomial

.. autoclass:: UQpy.Distributions.Poisson

Multivariate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.Distributions.Multinomial

.. autoclass:: UQpy.Distributions.MVNormal


List of Copula
---------------

.. autoclass:: UQpy.Distributions.Gumbel

.. autoclass:: UQpy.Distributions.Clayton

.. autoclass:: UQpy.Distributions.Frank


User-defined Distributions and Copula
---------------------------------------------------

Defining custom distributions in ``UQpy`` can be done by sub-classing the appropriate parent class. For example, to define the ``bivariate Rosenbrock`` distribution model that it is not part of the available ``scipy`` distribution models one can do:

    >>> from UQpy.Distributions import DistributionND
    >>> class Rosenbrock(DistributionND):
    >>>     def __init__(self, p=20.):
    >>>         super().__init__(p=p)
    >>>     def pdf(self, x):
    >>>         return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['p'])
    >>>     def log_pdf(self, x):
    >>>          return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['p']
    >>> dist = Rosenbrock(p=20)
    >>> print(hasattr(dist, 'pdf'))
        True
    >>> print(hasattr(dist, 'rvs'))
        False
    >>> print(hasattr(dist, 'update_params'))
        True
        
Custom copula can be similarly defined by subclassing the ``Copula`` class.

.. toctree::
    :maxdepth: 2



	
	