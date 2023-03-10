Stochastic Process
==================

The :py:mod:`.stochastic_process` module consists of classes and functions to generate samples of stochastic processes from
prescribed properties of the process (e.g. power spectrum, bispectrum and/or autocorrelation function). The existing
classes rely on stochastic expansions taking the following general form,

.. math:: A(x) = \sum_{i=1}^N \theta(\omega) \phi(x),

such that the process can be expressed in terms of a set of uncorrelated random variables, :math:`\theta(\omega)`, and
deterministic basis functions :math:`\phi(x)`.


The :py:mod:`.stochastic_process` module supports simulation of uni-variate, multi-variate, multi-dimensional, Gaussian
and non-Gaussian stochastic processes. Gaussian stochasitc processes can be simulated using the widely-used Spectral
Representation Method (:cite:`StochasticProcess1`, :cite:`StochasticProcess2`, :cite:`StochasticProcess3`, :cite:`StochasticProcess4`)
and the Karhunen-Loeve Expansion (:cite:`StochasticProcess5`, :cite:`StochasticProcess6`, :cite:`StochasticProcess7`). Non-Gaussian
stochastic processes can be generated through higher-order spectral representations (:cite:`StochasticProcess8`, :cite:`StochasticProcess9`,
:cite:`StochasticProcess10`) or through a
nonlinear transformation from a Gaussian stochastic process to a prescribed marginal distribution using translation
process theory :cite:`StochasticProcess11`. Modeling of arbitrarily distributed random processes with specified correlation and/or power
spectrum can be performed using the Iterative Translation Approximation Method (ITAM) (:cite:`StochasticProcess12`, :cite:`StochasticProcess13`) for inverse
translation process modeling.

This module contains functionality for all the stochastic process methods supported in UQpy.

The module currently contains the following classes:

- :class:`.SpectralRepresentation`: Class for simulation of Gaussian stochastic processes and random fields using the Spectral Representation Method.

- :class:`.BispectralRepresentation`: Class for simulation of third-order non-Gaussian stochastic processes and random fields using the Bispectral Representation Method.

- :class:`.KarhunenLoeveExpansion`: Class for simulation of stochastic processes using the Karhunen-Loeve Expansion.

- :class:`.Translation`: Class for transforming a Gaussian stochastic process to a non-Gaussian stochastic process with prescribed marginal probability distribution.

- :class:`.InverseTranslation`: Call for identifying an underlying Gaussian stochastic process for a non-Gaussian process with prescribed marginal probability distribution and autocorrelation function / power spectrum.

As with other modules of :py:mod:`.UQpy`, adding simulation methods requires the user to build a new class to support
the desired functionality. It does not require modification of any existing classes or methods.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Stochastic Processes

    Spectral Representation Method <spectral_representation>
    Bispectral Representation Method <bispectral_representation>
    Karhunen Loeve Expansion <karhunen_loeve_1d>
    Karhunen Loeve Expansion 2D <karhunen_loeve_2d>
    Translation Processes <translation>
