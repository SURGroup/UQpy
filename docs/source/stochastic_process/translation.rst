Non-Gaussian Translation Processes
-----------------------------------

A translation processes results from the nonlinear transformation of a Gaussian stochastic process. The standard
translation process, introduced and extensively studied by Grigoriu :cite:`StochasticProcess11` and implemented in :py:mod:`UQpy` arises from
the pointwise transformation of a Gaussian process through the inverse cumulative distribution function of a specified
marginal probability distribution as:

.. math:: X(t) = F^{-1}(\Phi(Y(t)))

where :math:`Y(x)` is a Gaussian random process with zero mean and unit standard deviation, :math:`\Phi(x)` is the
standard normal cumulative distribution function and :math:`F^{-1}(\cdot)` is the inverse cumulative distribution
function of the prescribed non-Gaussian probability distribution.

The nonlinear translation in the equation above has the effect of distorting the correlation of the stochastic process.
That is, if the Gaussian process has correlation function :math:`\rho(\tau)` where :math:`\tau=t_2-t_1` is a time lag,
then the non-Gaussian correlation function of the process :math:`X(t)` can be compuated as:

.. math:: \xi(\tau) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} F^{-1}(\Phi(y_1)) F^{-1}(\Phi(y_2)) \phi(y_1, y_2; \rho(\tau)) dy_1 dy_2

where :math:`\phi(y_1, y_2; \rho(\tau))` is the bivariate normal probability density function having correlation
:math:`\rho(\tau)`. This operation is known as correlation distortion and is not, in general, invertible. That is, given
the non-Gaussian correlation function :math:`\xi(\tau) ` and an arbitrarily prescribed non-Gaussian probability
distribution with cdf :math:`F(x)`, it is not always possible to identify a correponding Gaussian process having
correlation function :math:`\rho(\tau)` that can be translated to this non-Gaussian process through the equations
above :cite:`StochasticProcess11`.

This gives rise to the challenge of inverse translation process modeling, where the objective is to find the an
underlying Gaussian process and its correlation function such that it maps as closely as possible to the desired
non-Gaussian stochastic process with its arbitrarily prescribed distribution and correlation function. This problem is
solved in :py:mod:`UQpy` using the Iterative Translation Approximation Method (ITAM) developed in :cite:`StochasticProcess12` for processes
described by their power spectrum (and using :class:`.SpectralRepresentation` for simulation) and in :cite:`StochasticProcess13` for processes
described through their correlation function (and using :class:`.KarhunenLoeveExpansion` for simulation). This is implemented
in the :class:`InverseTranslation` class.

Translation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.stochastic_process.Translation
    :members:
    :private-members:

Inverse Translation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.stochastic_process.InverseTranslation
    :members:
    :private-members:
