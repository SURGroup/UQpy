Chatterjee indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These examples serve as a guide for using the Chatterjee sensitivity module. They have been taken from various papers to enable validation of the implementation and have been referenced accordingly.

1. **Ishigami function**

    In addition to the Pick and Freeze scheme, the Sobol indices can be estimated using the rank statistics approach [2]_. We demonstrate this estimation of the Sobol indices using the Ishigami function.

2. **Exponential function**

    For the Exponential model, analytical Cramér-von Mises indices are available [1]_ and since they are equivalent to the Chatterjee indices in the sample limit, they are shown here.

3. **Sobol function**

    This example was considered in [2]_ (page 18) to compare the Pick and Freeze scheme with the rank statistics approach for estimating the Sobol indices.

.. [1] Gamboa, F., Klein, T., & Lagnoux, A. (2018). Sensitivity Analysis Based on Cramér-von Mises Distance. SIAM/ASA Journal on Uncertainty Quantification, 6(2), 522-548. doi:10.1137/15M1025621. (`Link <https://doi.org/10.1137/15M1025621>`_)

.. [2] Fabrice Gamboa, Pierre Gremaud, Thierry Klein, and Agnès Lagnoux. (2020). Global Sensitivity Analysis: a new generation of mighty estimators based on rank statistics.
