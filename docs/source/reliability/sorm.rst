SORM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In SORM :cite:`FORM_XDu` the performance function is approximated by a second-order Taylor series around the design point according to


.. math:: G(\textbf{U}) = G(\textbf{U}^\star) +  \nabla G(\textbf{U}^\star)(\textbf{U}-\textbf{U}^\star)^\intercal + \frac{1}{2}(\textbf{U}-\textbf{U}^\star)\textbf{H}(\textbf{U}-\textbf{U}^\star)(\textbf{U}-\textbf{U}^\star)^T

where :math:`\textbf{H}` is the Hessian matrix of the second derivatives of :math:`G(\textbf{U})` evaluated at
:math:`\textbf{U}^*`. After the design point :math:`\textbf{U}^*` is identified and the probability of failure
:math:`P_{f, \text{form}}` is calculated with FORM a correction is made according to


.. math:: P_{f, \text{sorm}} = \Phi(-\beta_{HL}) \prod_{i=1}^{n-1} (1+\beta_{HL}\kappa_i)^{-\frac{1}{2}}

where :math:`\kappa_i` is the `i-th`  curvature.

The :class:`.SORM` class is imported using the following command:

>>> from UQpy.reliability.taylor_series.SORM import SORM

Methods
"""""""
.. autoclass:: UQpy.reliability.taylor_series.SORM
    :members: build_from_first_order

Attributes
""""""""""
.. autoattribute:: UQpy.reliability.taylor_series.SORM.beta_second_order
.. autoattribute:: UQpy.reliability.taylor_series.SORM.failure_probability

Examples
""""""""""

.. toctree::

   SORM Examples <../auto_examples/reliability/sorm/index>