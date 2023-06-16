FORM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


In FORM, the performance function is linearized according to

.. math:: G(\textbf{U})  \approx  G(\textbf{U}^\star) + \nabla G(\textbf{U}^\star)(\textbf{U}-\textbf{U}^\star)^\intercal

where :math:`\textbf{U}^\star` is the expansion point, :math:`G(\textbf{U})` is the performance function evaluated in
the standard normal space and :math:`\nabla G(\textbf{U}^\star)` is the gradient of :math:`G(\textbf{U})` evaluated at
:math:`\textbf{U}^\star`. The probability failure can be calculated by

.. math:: P_{f, \text{form}} = \Phi(-\beta_{HL})

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function and :math:`\beta_{HL}=||\textbf{U}^*||`
is the norm of the design point known as the Hasofer-Lind reliability index calculated with the iterative
Hasofer-Lind-Rackwitz-Fiessler (HLRF) algorithm.  The convergence criteria used for HLRF algorithm are:


.. math:: \text{tolerance}_U:\ ||\textbf{U}^{k} - \textbf{U}^{k-1}||_2 \leq 10^{-3}
.. math:: \text{tolerance}_\beta:\ ||\beta_{HL}^{k} - \beta_{HL}^{k-1}||_2 \leq 10^{-3}
.. math:: \text{tolerance}_{\nabla G(U)}:\ ||\nabla G(\textbf{U}^{k})- \nabla G(\textbf{U}^{k-1})||_2 \leq 10^{-3}



The :class:`.FORM` class is imported using the following command:

>>> from UQpy.reliability.taylor_series.FORM import FORM

Methods
"""""""
.. autoclass:: UQpy.reliability.taylor_series.FORM
    :members: run

Attributes
""""""""""

.. autoattribute:: UQpy.reliability.taylor_series.FORM.alpha

.. autoattribute:: UQpy.reliability.taylor_series.FORM.alpha_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.beta

.. autoattribute:: UQpy.reliability.taylor_series.FORM.beta_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.design_point_u

.. autoattribute:: UQpy.reliability.taylor_series.FORM.design_point_x

.. autoattribute:: UQpy.reliability.taylor_series.FORM.error_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.failure_probability

.. autoattribute:: UQpy.reliability.taylor_series.FORM.iterations

.. autoattribute:: UQpy.reliability.taylor_series.FORM.state_function_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.state_function_gradient_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.u_record

.. autoattribute:: UQpy.reliability.taylor_series.FORM.x_record


Examples
""""""""""

.. toctree::

   FORM Examples <../auto_examples/reliability/form/index>
