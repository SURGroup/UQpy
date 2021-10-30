StochasticProcess
=================

The :py:mod:`.stochastic_process` module consists of classes and functions to generate samples of stochastic processes from
prescribed properties of the process (e.g. power spectrum, bispectrum and/or autocorrelation function). The existing
classes rely on stochastic expansions taking the following general form,

.. math:: A(x) = \sum_{i=1}^N \theta(\omega) \phi(x),

such that the process can be expressed in terms of a set of uncorrelated random variables, :math:`\theta(\omega)`, and
deterministic basis functions :math:`\phi(x)`.


The :py:mod:`.stochastic_process` module supports simulation of uni-variate, multi-variate, multi-dimensional, Gaussian
and non-Gaussian stochastic processes. Gaussian stochasitc processes can be simulated using the widely-used Spectral
Representation Method ([1]_, [2]_, [3]_, [4]_) and the Karhunen-Loeve Expansion ([5]_, [6]_, [7]_). Non-Gaussian
stochastic processes can be generated through higher-order spectral representations ([8]_, [9]_, [10]_) or through a
nonlinear transformation from a Gaussian stochastic process to a prescribed marginal distribution using translation
process theory [11]_. Modeling of arbitrarily distributed random processes with specified correlation and/or power
spectrum can be performed using the Iterative Translation Approximation Method (ITAM) ([12]_, [13]_) for inverse
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
    Karhunen Loeve Expansion <karhunen_loeve>
    Non-Gaussian Translation Processes <translation>



.. [1] Shinozuka, M. and Jan, C. M. (1972). "Digital simulation of random processes and its applications," Journal of Sound and Vibration. 25: 111–128.

.. [2] Shinozuka, M. and Deodatis, G. (1991) "Simulation of Stochastic Processes by Spectral representation" Applied Mechanics Reviews. 44.

.. [3] Shinozuka, M. and Deodatis, G. (1996) "Simulation of multi-dimensional Gaussian stochastic fields by spectral representation," Applied Mechanics Reviews. 49: 29–53.

.. [4] Deodatis, G. "Simulation of Ergodic Multivariate Stochastic Processes," Journal of Engineering Mechanics. 122: 778–787.

.. [5] Huang, S.P., Quek, S. T., and Phoon, K. K. (2001). "Convergence study of the truncated Karhunen-Loeve expansion for simulation of stochastic processes," International Journal for Numerical Methods in Engineering. 52: 1029–1043.

.. [6] Phoon K.K., Huang S.P., Quek S.T. (2002). "Simulation of second-order processes using Karhunen–Loève expansion." Computers and Structures 80(12):1049–60.

.. [7] Betz, W., Papaioannou, I., & Straub, D. (2014). "Numerical methods for the discretization of random fields by means of the Karhunen–Loève expansion." Computer Methods in Applied Mechanics and Engineering, 271: 109-129.

.. [8] Shields, M.D. and Kim, H. (2017). "Simulation of higher-order stochastic processes by spectral representation," Probabilistic Engineering Mechanics. 47: 1-15.

.. [9] Vandanapu, L. and Shields, M.D. (2020). "3rd-order Spectral Representation Methods: Multi-dimensional random fields with fast Fourier transform implementation." arXiv:1910.06420

.. [10] Vandanapu, L. and Shields, M.D. (2020). "3rd-order Spectral Representation Methods: Ergodic multi-variate random processes with fast Fourier transform." arXiv:1910.06420

.. [11] Grigoriu, M. (1995). "Applied Non-Gaussian Processes," Pretice Hall.

.. [12] Shields, M.D., Deodatis, G., and Bocchini, P. (2011). "A simple and efficient methodology to approximate a general non-Gaussian stationary stochastic process by a translation process," Probabilistic Engineering Mechanics. 26: 511-519.

.. [13] Kim, H. and Shields, M.D. (2105). "Modeling strongly non-Gaussian non-stationary stochastic processes using the Iterative Translation Approximation Method and Karhunen–Loève expansion," Computers and Structures. 161: 31-42.
