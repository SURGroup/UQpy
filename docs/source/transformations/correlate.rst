Correlate
-----------------

:class:`.Correlate` is a class to induce correlation to an uncorrelated standard normal vector :math:`\textbf{u}=[U_1,...,U_n]`,
given the correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The correlated standard normal vector
:math:`\textbf{z}=[Z_1,...,Z_n]` can be calculated as:

.. math:: \mathbf{z}^\intercal = \mathbf{H}\mathbf{u}^\intercal

where :math:`\mathbf{H}` is the lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix, i.e. :math:`\mathbf{C_z}=\mathbf{H}\mathbf{H}^\intercal`.

Correlate Class
^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.transformations.Correlate

Attributes
""""""""""

.. autoattribute:: UQpy.transformations.Correlate.H
.. autoattribute:: UQpy.transformations.Correlate.samples_z