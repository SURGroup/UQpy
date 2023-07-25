Karhunen Loève Expansion for Multi-Dimensional Fields
-----------------------------------------------------

The Karhunen Loève Expansion expands the stochastic field as follows:

.. math:: A(x, t) = \sum_{n=1}^{\infty} \sum_{k=1}^{\infty}\eta_{nk}(\theta) \sqrt{\lambda_n(x)} f_n(t, x) \sqrt{\mu_{nk}} g_{nk}(x)

where :math:`\eta_{nk}(\theta)` are uncorrelated standardized normal random variables and :math:`\lambda_n(x)` and :math:`f_n(x, t)` are the eigenvalues and eigenvectors repsectively of the "quasi" one dimensional covariance function :math:`C(x, t_1, t_2)`. :math:`\mu_{nk}` and :math:`g_{nk}(x)` are the eigenvalues and eigenvectors of the derived "one" dimensional covariance function :math:`H(x_1, x_2)`. Additional details regarding the simulation formula can be found
at :cite:`Kle2D`

KarhunenLoeve2D Class
^^^^^^^^^^^^^^^^^^^^^

The :class:`.KarhunenLoeve2D` class is imported using the following command:

>>> from UQpy.stochastic_process.KarhunenLoeveExpansionTwoDimension2D import KarhunenLoeveExpansion

Methods
"""""""
.. autoclass:: UQpy.stochastic_process.KarhunenLoeveExpansion2D
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.stochastic_process.KarhunenLoeveExpansion2D.samples

Examples
""""""""""

.. toctree::

   Karhunen Loeve Examples <../auto_examples/stochastic_processes/karhunen_loeve_2d/index>