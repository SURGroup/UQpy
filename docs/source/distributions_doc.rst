.. _distributions_doc:

Distributions
=============

.. automodule:: UQpy.distributions

Note that the various classes of the ``Distributions`` module are written to be consistent with distribuitons in the
``scipy.stats`` package [1]_, to the extent possible while maintaining an extensible, object oriented architecture that is
convenient for operating with the other ``UQpy`` modules. All existing distributions and their methods in ``UQpy`` are restructured from the ``scipy.stats`` package. 


Parent Distribution Class
----------------------------------

.. autoclass:: UQpy.distributions.baseclass.Distribution
    :members:


1D Continuous Distributions
---------------------------------------

In ``UQpy``, univariate continuous distributions inherit from the ``DistributionContinuous1D`` class:

.. autoclass:: UQpy.distributions.baseclass.DistributionContinuous1D
    :members: 

List of 1D Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is a list of all 1D continuous distributions currently available in ``UQpy``.

.. autoclass:: UQpy.distributions.collections.Beta

.. autoclass:: UQpy.distributions.collection.Cauchy

.. autoclass:: UQpy.distributions.collection.ChiSquare

.. autoclass:: UQpy.distributions.collection.Exponential

.. autoclass:: UQpy.distributions.collection.Gamma

.. autoclass:: UQpy.distributions.collection.GenExtreme

.. autoclass:: UQpy.distributions.collection.InvGauss

.. autoclass:: UQpy.distributions.collection.Laplace

.. autoclass:: UQpy.distributions.collection.Levy

.. autoclass:: UQpy.distributions.collection.Logistic

.. autoclass:: UQpy.distributions.collection.Lognormal

.. autoclass:: UQpy.distributions.collection.Maxwell

.. autoclass:: UQpy.distributions.collection.Normal

.. autoclass:: UQpy.distributions.collection.Pareto

.. autoclass:: UQpy.distributions.collection.Rayleigh

.. autoclass:: UQpy.distributions.collection.TruncNorm

.. autoclass:: UQpy.distributions.collection.Uniform

1D Discrete Distributions
----------------------------------

In ``UQpy``, univariate discrete distributions inherit from the ``DistributionDiscrete1D`` class:

.. autoclass:: UQpy.distributions.baseclass.DistributionDiscrete1D
    :members: 

List of 1D Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is a list of all 1D discrete distributions currently available in ``UQpy``.

.. autoclass:: UQpy.distributions.collection.Binomial

.. autoclass:: UQpy.distributions.collection.Poisson

Multivariate Distributions
----------------------------------

In ``UQpy``, multivariate distributions inherit from the ``DistributionND`` class:

.. autoclass:: UQpy.distributions.baseclass.DistributionND


``UQpy`` has some inbuilt multivariate distributions, which are directly child classes of ``DistributionND``. Additionally, joint distributions can be built from their marginals through the use of the ``JointIndependent`` and ``JointCopula`` classes described below.

List of Multivariate Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.distributions.collection.Multinomial

.. autoclass:: UQpy.distributions.collection.MultivariateNormal
    
Joint from independent marginals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.distributions.collection.JointIndependent

Joint from marginals and copula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.distributions.collection.JointCopula


Copula
-----------

.. autoclass:: UQpy.distributions.baseclass.Copula



List of Copulas
~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.distributions.copulas.Gumbel

.. autoclass:: UQpy.distributions.copulas.Clayton

.. autoclass:: UQpy.distributions.copulas.Frank


User-defined Distributions and Copulas
---------------------------------------------------

Defining custom distributions in ``UQpy`` can be done by sub-classing the appropriate parent class. The subclasses must possess the desired methods, per the parent ``Distribution`` class. 

Custom copulas can be similarly defined by subclassing the ``Copula`` class and defining the appropriate methods.

.. toctree::
    :maxdepth: 2

.. [1] https://docs.scipy.org/doc/scipy/reference/stats.html


