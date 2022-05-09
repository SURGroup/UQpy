Cramér-von Mises Sensitivity indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These examples serve as a guide for using the Cramér-von Mises sensitivity module. They have been taken from various papers to enable validation of the implementation and have been referenced accordingly.

1. **Exponential function**

    For the Exponential model, analytical Cramér-von Mises indices are available [1]_.

2. **Sobol function**

    The Cramér-von Mises indices are computed using the Pick and Freeze approach [1]_. These model evaluations can be used to estimate the Sobol indices as well. We demonstrate this using the Sobol function.

.. [1] Gamboa, F., Klein, T., & Lagnoux, A. (2018). Sensitivity Analysis Based on Cramér-von Mises Distance. SIAM/ASA Journal on Uncertainty Quantification, 6(2), 522-548. doi:10.1137/15M1025621. (`Link <https://doi.org/10.1137/15M1025621>`_)