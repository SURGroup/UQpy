List of Multivariate Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Multinomial
"""""""""""

Multinomial distribution having probability mass function

.. math:: f(x) = \dfrac{n!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}

for :math:`x=\{x_1,\dots,x_k\}` where each :math:`x_i` is a non-negative integer and :math:`\sum_i x_i = n`.

The :class:`.Multinomial` class is imported using the following command:

>>> from UQpy.distributions.collection.Multinomial import Multinomial

.. autoclass:: UQpy.distributions.collection.Multinomial
    :members:

The following methods are available for :class:`.Multinomial`:

:py:meth:`pmf`, :py:meth:`log_pmf`, :py:meth:`rvs`, :py:meth:`moments` .

______

Multivariate Normal
"""""""""""""""""""

Multivariate normal distribution having probability density function

.. math:: f(x) = \dfrac{1}{\sqrt{(2\pi)^k\det\Sigma}}\exp{-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}

where :math:`\mu` is the mean vector, :math:`\Sigma` is the covariance matrix, and :math:`k` is the dimension of
`x`.

The :class:`.MultivariateNormal` class is imported using the following command:

>>> from UQpy.distributions.collection.MultivariateNormal import MultivariateNormal

.. autoclass:: UQpy.distributions.collection.MultivariateNormal
    :members:

The following methods are available for :class:`.MultivariateNormal` :

:py:meth:`pdf`, :py:meth:`log_pdf`, :py:meth:`rvs`, :py:meth:`fit`, :py:meth:`moments`.

