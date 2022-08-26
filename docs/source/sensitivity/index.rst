Sensitivity
===============

This module contains functionality for all the sampling methods supported in :py:mod:`UQpy`.

The module currently contains the following classes:

- :py:class:`.Chatterjee`: Class to compute Chatterjee sensitivity indices.
- :py:class:`.CramervonMises`: Class to compute Cramér-von Mises sensitivity indices.
- :py:class:`.GeneralisedSobol`: Class to compute Generalised Sobol sensitivity indices.
- :py:class:`.MorrisSensitivity`: Class to perform Morris.
- :py:class:`.PceSensitivity`: Class to compute the sensitivity indices using the :class:`.PolynomialChaosExpansion` method.
- :py:class:`.Sobol`: Class to compute Sobol sensitivity indices.

Sensitivity analysis comprises techniques focused on determining how the variations of input variables :math:`X=\left[ X_{1}, X_{2},…,X_{d} \right]` of a mathematical model influence the response value :math:`Y=h(X)`.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Sensitivity

    Chatterjee Sensitivity <chatterjee>
    Cramér-von Mises Sensitivity <cramer_von_mises>
    Generalised Sobol Sensitivity <generalised_sobol>
    Morris Sensitivity <morris>
    Polynomial Chaos Sensitivity <pce>
    Sobol Sensitivity <sobol>

Examples
""""""""""

.. toctree::

   Comparison of indices <../auto_examples/sensitivity/comparison/index>
