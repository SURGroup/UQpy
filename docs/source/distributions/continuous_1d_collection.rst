List of 1D Continuous Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a list of all 1D continuous distributions currently available in :py:mod:`UQpy`.

Beta
""""

Beta distribution having probability density function

.. math:: f(x|a,b) = \dfrac{ \Gamma(a+b)x^{a-1}(1-x)^{b-1}}{\Gamma(a) \Gamma(b)}

for :math:`0 \le x \ge 0`, :math:`a > 0, b > 0`. Here :math:`\Gamma (a)` refers to the Gamma function.

In this standard form :math:`(loc=0, scale=1)`, the distribution is defined over the interval (0, 1). Use `loc` and
`scale` to shift the distribution to interval :math:`(loc, loc + scale)`. Specifically, this is equivalent to computing
:math:`f(y|a,b)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Beta

______

Cauchy
""""""

Cauchy distribution having probability density function

.. math:: f(x) = \dfrac{1}{\pi(1+x^2)}

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Cauchy

__________

Chi Square
""""""""""

Chi-square distribution having probability density:

.. math:: f(x|k) = \dfrac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}\exp{(-x/2)}

for :math:`x\ge 0`, :math:`k>0`. Here :math:`\Gamma(\cdot)` refers to the Gamma function.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y|k)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.ChiSquare

___________

Exponential
"""""""""""

Exponential distribution having probability density function:

.. math:: f(x) = \exp(-x)

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

A common parameterization for Exponential is in terms of the rate parameter :math:`\lambda`, which corresponds to
using :math:`scale = 1 / \lambda`.

.. autoclass:: UQpy.distributions.collection.Exponential

___________

Gamma
"""""""""""

Gamma distribution having probability density function:

.. math:: f(x|a) = \dfrac{x^{a-1}\exp(-x)}{\Gamma(a)}
for :math:`x\ge 0`, :math:`a>0`. Here :math:`\Gamma(a)` refers to the Gamma function.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Gamma

___________________

Generalized Extreme
"""""""""""""""""""

Generalized Extreme Value distribution having probability density function:

.. math:: f(x|c) = \exp(-(1-cx)^{1/c})(1-cx)^{1/c-1}

for :math:`x\le 1/c, c>0`.

For :math:`c=0`

.. math:: f(x) = \exp(\exp(-x))\exp(-x)

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.GeneralizedExtreme

________________

Inverse Gaussian
""""""""""""""""

Inverse Gaussian distribution having probability density function

.. math:: f(x|\mu) = \dfrac{1}{2\pi x^3}\exp{(-\dfrac{(x\\mu)^2}{2x\mu^2})}

for :math:`x>0`. :py:meth:`cdf` method returns :any:`NaN` for :math:`\mu<0.0028`.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.InverseGauss

_______

Laplace
"""""""

Laplace distribution having probability density function

.. math:: f(x) = \dfrac{1}{2}\exp{-|x|}

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Laplace

____

Levy
""""

Levy distribution having probability density function

.. math:: f(x) = \dfrac{1}{\sqrt{2\pi x^3}}\exp(-\dfrac{1}{2x})

for :math:`x\ge 0`.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Levy

________

Logistic
""""""""

Logistic distribution having probability density function

.. math:: f(x) = \dfrac{\exp(-x)}{(1+\exp(-x))^2}

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Logistic

_________

Lognormal
"""""""""

Lognormal distribution having probability density function

.. math:: f(x|s) = \dfrac{1}{sx\sqrt{2\pi}}\exp(-\dfrac{\log^2(x)}{2s^2})

for :math:`x>0, s>0`.

A common parametrization for a lognormal random variable :math:`Y` is in terms of the mean, mu, and standard deviation,
sigma, of the gaussian random variable :math:`X` such that :math:`exp(X) = Y`. This parametrization corresponds to setting
:math:`s = sigma` and :math:`scale = exp(mu)`.

.. autoclass:: UQpy.distributions.collection.Lognormal

_______

Maxwell
"""""""

Maxwell-Boltzmann distribution having probability density function

.. math:: f(x) = \sqrt{2/\pi}x^2\exp(-x^2/2)

for :math:`x\ge0`.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Maxwell

_______

Normal
"""""""

Normal distribution having probability density function

.. math:: f(x) = \dfrac{\exp(-x^2/2)}{\sqrt{2\pi}}

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Normal

_______

Pareto
"""""""

Pareto distribution having probability density function

.. math:: f(x|b) = \dfrac{b}{x^{b+1}}

for :math:`x\ge 1, b>0`.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Pareto

_______

Rayleigh
""""""""

Rayleigh distribution having probability density function

.. math:: f(x) = x\exp(-x^2/2)

for :math:`x\ge 0`.

In this standard form :math:`(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

.. autoclass:: UQpy.distributions.collection.Rayleigh

_______

Truncated Normal
""""""""""""""""

Truncated normal distribution

The standard form of this distribution :math:`(loc=0, scale=1)` is a standard normal truncated to the range :math:`[a, b]`.
Note that *a* and *b* are defined over the domain of the standard normal.

.. autoclass:: UQpy.distributions.collection.TruncatedNormal

_______

Uniform
"""""""

Uniform distribution having probability density function

.. math:: f(x|a, b) = \dfrac{1}{b-a}

where :math:`a=loc` and :math:`b=loc+scale`

.. autoclass:: UQpy.distributions.collection.Uniform