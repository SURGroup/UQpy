Copula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copula Class
^^^^^^^^^^^^^^^^

Methods
~~~~~~~~~~~~~~~~~~
.. autoclass:: UQpy.distributions.baseclass.Copula
    :members: get_parameters, update_parameters, check_marginals

Attributes
~~~~~~~~~~~~~~~~~~
.. autoattribute:: UQpy.distributions.baseclass.Copula.parameters
.. autoattribute:: UQpy.distributions.baseclass.Copula.ordered_parameters



----

List of Copulas
^^^^^^^^^^^^^^^^


Clayton
~~~~~~~~~~~~~~~~~~

Clayton copula having cumulative distribution function

.. math:: F(u_1, u_2) = \max(u_1^{-\Theta} + u_2^{-\Theta} - 1, 0)^{-1/{\Theta}}

where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

.. autoclass:: UQpy.distributions.copulas.Clayton
    :members:

______

Frank
~~~~~~~~~~~~~~~~~~

Frank copula having cumulative distribution function

:math:`F(u_1, u_2) = -\dfrac{1}{\Theta} \log(1+\dfrac{(\exp(-\Theta u_1)-1)(\exp(-\Theta u_2)-1)}{\exp(-\Theta)-1})`

where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.


.. autoclass:: UQpy.distributions.copulas.Frank
    :members:

______

Gumbel
~~~~~~~~~~~~~~~~~~

Gumbel copula having cumulative distribution function

.. math:: F(u_1, u_2) = \exp(-(-\log(u_1))^{\Theta} + (-\log(u_2))^{\Theta})^{1/{\Theta}}

where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

.. autoclass:: UQpy.distributions.copulas.Gumbel
    :members: