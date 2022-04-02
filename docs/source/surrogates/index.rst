Surrogates
==========

This module contains functionality for all the surrogate methods supported in UQpy.

The module currently contains the following classes:

- :class:`.SROM`: Class to estimate a discrete approximation for a continuous random variable using Stochastic Reduced Order Model.

- :class:`.GaussianProcessRegressor`: Class to generate an approximate surrogate model using Gaussian Processes.

- :class:`.PolynomialChaosExpansion`: Class to generate an approximate surrogate model using Polynomial chaos.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Surrogates

    Stochastic Reduced Order Models <srom>
    Gaussian Process Regression <gpr>
    Polynomial Chaos Expansion <polynomial_chaos>