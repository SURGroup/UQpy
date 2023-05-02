Theta Criterion
---------------
The technique enables one-by-one extension of an experimental design while trying to obtain an optimal sample at each stage of the adaptive sequential surrogate model
construction process. The sequential sampling strategy based on :math:`\Theta` criterion selects from a pool of candidate points by trying to cover the design domain
proportionally to their local variance contribution. The proposed criterion for the sample selection balances both exploitation of the surrogate model using variance
density derived analytically from Polynomial Chaos Expansion and exploration of the design domain. The active learning technique based on  :math:`\Theta` criterion can be
combined with arbitrary sampling technique employed for construction of a pool of candidate points. More details can be found in:

L. Novák, M. Vořechovský, V. Sadílek, M. D. Shields, *Variance-based adaptive sequential sampling for polynomial chaos expansion*,
637 Computer Methods in Applied Mechanics and Engineering 386 (2021) 114105. doi:10.1016/j.cma.2021.114105


ThetaCriterionPCE Class
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ThetaCriterionPCE` class is imported using the following command:

>>> from UQpy.sampling.ThetaCriterionPCE import ThetaCriterionPCE


Methods
"""""""""""
.. autoclass:: UQpy.sampling.ThetaCriterionPCE
    :members: run


Examples
"""""""""""

.. toctree::

    Theta Criterion Examples <../auto_examples/sampling/theta_criterion/index>
