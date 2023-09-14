Inverse FORM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In FORM :cite:`FORM_XDu`, the performance function is linearized according to

.. math:: G(\textbf{U})  \approx  G(\textbf{U}^\star) + \nabla G(\textbf{U}^\star)(\textbf{U}-\textbf{U}^\star)^\intercal

where :math:`\textbf{U}^\star` is the expansion point, :math:`G(\textbf{U})` is the performance function evaluated in
the standard normal space and :math:`\nabla G(\textbf{U}^\star)` is the gradient of :math:`G(\textbf{U})` evaluated at
:math:`\textbf{U}^\star`. The probability failure is approximated as

.. math:: p_{fail} = \Phi(-\beta_{HL})

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function and :math:`\beta_{HL}=||\textbf{U}^*||`
is the norm of the design point known as the Hasofer-Lind reliability index.

The goal of the inverse FORM algorithm is to find a design point :math:`\textbf{U}^\star` that minimizes the performance
function :math:`G(\textbf{U})` and lies on the hypersphere defined by :math:`||\textbf{U}^*|| = \beta_{HL}`, or
equivalently :math:`||\textbf{U}^*|| = -\Phi^{-1}(p_{fail})`. The default convergence criteria used for this algorithm
are:

.. math:: \text{tolerance}_{\textbf{U}}:\ ||\textbf{U}_i - \textbf{U}_{i-1}||_2 \leq 10^{-3}
.. math:: \text{tolerance}_{\nabla G(\textbf{U})}:\ ||\nabla G(\textbf{U}_i)- \nabla G(\textbf{U}_{i-1})||_2 \leq 10^{-3}


**Problem Statement**

Compute :math:`u^* = \text{argmin}\ G(\textbf{U})` such that :math:`||\textbf{U}||=\beta`.

The feasibility criteria :math:`||\textbf{U}||=\beta` may be equivalently defined as
:math:`\beta = -\Phi^{-1}(p_{fail})`, where :math:`\Phi^{-1}(\cdot)` is the inverse standard normal CDF.

**Algorithm**

This method implements a gradient descent algorithm to solve the optimization problem within the tolerances specified by
:code:`tolerance_u` and :code:`tolerance_gradient`.

0. Define :math:`u_0` and :math:`\beta` directly or :math:`\beta=-\Phi^{-1}(p_\text{fail})`
1. Set :math:`u=u_0` and :math:`\text{converged}=False`
2. While not :math:`\text{converged}`:
    a. :math:`\alpha = \frac{\nabla G(u)}{|| \nabla G(u) ||}`
    b. :math:`u_{new} = - \alpha \beta`
    c. If :math:`||u_{new} - u || \leq \text{tolerance}_u` and/or :math:`|| \nabla G(u_{new}) - \nabla G(u) || \leq \text{tolerance}_{\nabla G(u)}`;
        set :math:`\text{converged}=True`
       else;
        :math:`u = u_{new}`

The :class:`.InverseFORM` class is imported using the following command:

>>> from UQpy.reliability.taylor_series import InverseFORM

Methods
-------

.. autoclass:: UQpy.reliability.taylor_series.InverseFORM
    :members: run

Attributes
----------

.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.alpha
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.alpha_record
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.beta
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.beta_record
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.design_point_u
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.design_point_x
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.error_record
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.iteration_record
.. autoattribute:: UQpy.reliability.taylor_series.InverseFORM.failure_probability_record


Examples
--------

.. toctree::

   InverseFORM Examples <../auto_examples/reliability/inverse_form/index>