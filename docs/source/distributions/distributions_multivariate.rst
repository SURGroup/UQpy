Multivariate Distributions
----------------------------------

In :py:mod:`UQpy`, multivariate distributions inherit from the :class:`.DistributionND` class


:py:mod:`UQpy` has some inbuilt multivariate distributions, which are directly child classes of :class:`.DistributionND`.
Additionally, joint distributions can be built from their marginals through the use of the :class:`.JointIndependent` and
:class:`.JointCopula` classes described below.

List of Multivariate Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

______

Multinomial
"""""""""""

Multinomial distribution having probability mass function

.. math:: f(x) = \dfrac{n!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}

for :math:`x=\{x_1,\dots,x_k\}` where each :math:`x_i` is a non-negative integer and :math:`\sum_i x_i = n`.

.. autoclass:: UQpy.distributions.collection.Multinomial
    :members:

______

Multivariate Normal
"""""""""""""""""""

Multivariate normal distribution having probability density function

.. math:: f(x) = \dfrac{1}{\sqrt{(2\pi)^k\det\Sigma}}\exp{-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}

where :math:`\mu` is the mean vector, :math:`\Sigma` is the covariance matrix, and :math:`k` is the dimension of
`x`.

.. autoclass:: UQpy.distributions.collection.MultivariateNormal
    :members:

Joint from independent marginals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a joint distribution from its independent marginals. :class:`.JointIndependent` is a child class of
:class:`.DistributionND`.

Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
also possess them:

* ``pdf``, ``log_pdf``, ``cdf``, ``rvs``, ``fit``, ``moments``.

The parameters of the distribution are only stored as attributes of the marginal objects. However, the
*get_parameters* and *update_parameters* method can still be used for the joint. Note that, for this purpose, each
parameter of the joint is assigned a unique string identifier as `key_index` - where `key` is the parameter name and
`index` the index of the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

.. autoclass:: UQpy.distributions.collection.JointIndependent
    :members:


Joint from marginals and copula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a joint distribution from a list of marginals and a copula to introduce dependency. :class:`.JointCopula` is a
child class of :class:`.DistributionND`.

A :class:`.JointCopula`` distribution may possess a ``cdf``, ``pdf`` and ``log_pdf`` methods if the copula allows for it
(i.e., if the copula possesses the necessary :meth:`evaluate_cdf` and :meth`.evaluate_pdf` methods).

The parameters of the distribution are only stored as attributes of the marginals/copula objects. However, the
:meth:`get_parameters` and :meth:`update_parameters` methods can still be used for the joint. Note that each parameter of
the joint is assigned a unique string identifier as `key_index` - where `key` is the parameter name and `index` the
index of the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`); and `key_c` for
copula parameters.

.. autoclass:: UQpy.distributions.collection.JointCopula
    :members:
