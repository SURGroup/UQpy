Karhunen Loève Expansion
----------------------------

The Karhunen Loève Expansion expands the stochastic process as follows:

.. math:: A(x) = \sum_{i=1}^N \sqrt{\lambda_i} \theta_i(\omega)f_i(x)

where :math:`\theta_i(\omega)` are uncorrelated standardized random variables and :math:`\lambda_i` and :math:`f_i(x)` are the eigenvalues and eigenvectors repsectively of the covariance function :math:`C(x_1, x_2)`.

KarhunenLoeve Class
^^^^^^^^^^^^^^^^^^^^

The :class:`.KarhunenLoaeve` class is imported using the following command:

>>> from UQpy.stochastic_process.KarhunenLoeveExpansion import KarhunenLoeveExpansion

Methods
"""""""
.. autoclass:: UQpy.stochastic_process.KarhunenLoeveExpansion
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.stochastic_process.KarhunenLoeveExpansion.samples
.. autoattribute:: UQpy.stochastic_process.KarhunenLoeveExpansion.xi

Examples
""""""""""

.. toctree::

   Karhunen Loeve Examples <../auto_examples/stochastic_processes/karhunen_loeve/index>