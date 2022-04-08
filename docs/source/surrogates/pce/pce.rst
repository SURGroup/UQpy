PolynomialChaosExpansion Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.PolynomialChaosExpansion` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion

Methods
"""""""
.. autoclass:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion
    :members: fit, predict, validation_error, leaveoneout_error, get_moments

Attributes
""""""""""
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.polynomial_basis
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.multi_index_set
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.coefficients
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.bias
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.outputs_number
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.design_matrix
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.experimental_design_input
.. autoattribute:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion.experimental_design_output

Examples
""""""""""

.. toctree::

   Polynomial Chaos Expansion Examples <../../auto_examples/surrogates/pce/index>

