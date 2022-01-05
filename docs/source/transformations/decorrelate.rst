Decorrelate
-----------------

:class:`.Decorrelate` is a class to remove correlation from an correlated standard normal vector
:math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The uncorrelated standard
normal vector :math:`\textbf{u}=[U_1,...,U_n]` can be calculated as:

.. math:: \textbf{u}^\intercal = \mathbf{H}^{-1} \mathbf{z}^\intercal

Decorrelate Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.transformations.Decorrelate


Attributes
""""""""""
.. autoattribute:: UQpy.transformations.Decorrelate.H
.. autoattribute:: UQpy.transformations.Decorrelate.samples_u