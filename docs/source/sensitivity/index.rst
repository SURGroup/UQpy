Sensitivity
===============

This module contains functionality for all the sampling methods supported in :py:mod:`UQpy`.

The module currently contains the following classes:

- :py:class:`.Sobol`: Class to compute Sobol sensitivity indices.
- :py:class:`.MorrisSensitivity`: Class to perform Morris.
- :py:class:`.PceSensitivity`: Class to compute the sensitivity indices using the :class:`.PolynomialChaosExpansion` method.

Sensitivity analysis comprises techniques focused on determining how the variations of input variables :math:`X=\left[ X_{1}, X_{2},…,X_{d} \right]` of a mathematical model influence the response value :math:`Y=h(X)`.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Sensitivity

    Morris Sensitivity <morris>
    Polynomial Chaos Sensitivity <pce>
    Sobol Sensitivity <sobol>
