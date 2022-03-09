Taylor Series Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first-order reliability methods (FORM), as stated by its name, utilizes a first-order Taylor
series expansions for the performance function in a standard normal probability space to derive
probability of failure estimates. Consider a model in standard normal space  with performance
function :math:`g(\textbf{U})`. The FORM approximates the performance function by:

.. math:: g(\textbf{U})  \approx L(\textbf{U}) = g(\textbf{u}^\star) + \nabla g(\textbf{u}^\star)(\textbf{U}-\textbf{u}^\star)^T


where :math:`\textbf{u}^\star` is the point around which the series is expanded and its typically
called the design point (needs to be found) and it corresponds to the point on the line
:math:`g(\textbf{U})=0` with the highest probability. :math:`\nabla g(\textbf{u}^\star)` is the gradient of
:math:`g(\textbf{U})` evaluated at :math:`\textbf{u}^\star`.
