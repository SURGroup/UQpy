List of Discrete Distributions 1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a list of all 1D discrete distributions currently available in :py:mod:`UQpy`.

______

Binomial
""""""""

Binomial distribution having probability mass function:

.. math:: f(x) = {n \choose x} p^x(1-p)^{n-x}

for :math:`x \in \{0, 1, 2, ..., n\}`.

In this standard form :math:`(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
:math:`f(y)` where :math:`y=x-loc`.

.. autoclass:: UQpy.distributions.collection.Binomial

______

Poisson
""""""""

Poisson distribution having probability mass function:

.. math:: f(x) = \exp{(-\mu)}\dfrac{\mu^k}{k!}

for :math:`x\ge 0`.

In this standard form :math:`(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
:math:`f(y)` where :math:`y=x-loc`.

.. autoclass:: UQpy.distributions.collection.Poisson