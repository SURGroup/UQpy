
Sobol indices
----------------------------------------

Sobol indices are the standard approach for performing a global sensitivity analysis. 
The indices are based on a variance decomposition of the model output. Using this decomposition allows us to assign the contribution of uncertain inputs to the variance of the model output.

There are three main groups of indices:

- First order indices (:math:`S_{i}`): Describe the fraction of the output variance due to a single uncertain input parameter :math:`i`. This amount of variance can be reduced if the uncertainty in the corresponding input is eliminated.

- Higher order indices: Describe the fraction of the output variance due to interactions between uncertain input parameters. For example, the second order indices (:math:`S_{ij}`) describe the fraction of the output variance due to interactions between two uncertain input parameters :math:`i` and :math:`j`.

- Total order indices (:math:`S_{T_{i}}`): Describe the fraction of the output variance due to a single input parameter :math:`i` and all higher order effects of the input parameter.

If the first order index of an input parameter is equal to the total order index it implies that the parameter does not have any interaction effects.

The Sobol indices are typically computed using the Pick-and-Freeze approach for single output and multi-output models. Since there are several variants of the Pick-and-Freeze approach, the schemes implemented to compute Sobol indices are listed below:

Here, :math:`N` is the Monte Carlo sample size and :math:`m` is the number of input parameters in the model.

1. **First order indices** (:math:`S_{i}`)

- Janon2014: Requires :math:`N(m + 1)` model evaluations

.. math:: 
   \frac{\mathbb{V}\left[E\left(Y \mid X_{i}\right)\right]}{\mathbb{V}(Y)} = \frac{\operatorname{Cov}\left(Y, Y_{C_{i}}\right)}{\mathbb{V}(Y)} = \frac{ (1 / N) Y_{A} \cdot Y_{C_{i}}-f_{0}^{2}}{ (1 / N)\frac{Y_{A} \cdot Y_{A} + Y_{C_{i}} \cdot Y_{C_{i}}}{2}-f_{0}^{2}}

.. math:: 
   y_{A}=f(A), \quad y_{C_{i}}=f(C_{i}), \quad f_{0}^{2}=\left(\frac{1}{2N} \sum_{j=1}^{N} y_{A}^{(j)} + y_{C_{i}}^{(j)} \right)^{2}

Compared to "Sobol1993", the "Janon2014" estimator makes more efficient use of model evaluations and produces smaller (better) confidence intervals.

- Sobol1993: Requires :math:`N(m + 1)` model evaluations :cite:`saltelli_2008`.

.. math::
   S_{i} = \frac{\mathbb{V}\left[E\left(Y \mid X_{i}\right)\right]}{\mathbb{V}(Y)} = \frac{ (1/N) Y_{A} \cdot Y_{C_{i}}-f_{0}^{2}}{(1 / N) Y_{A} \cdot Y_{A}-f_{0}^{2}}

.. math:: 
   y_{A}=f(A), \quad y_{C_{i}}=f(C_{i}), \quad f_{0}^{2}=\left(\frac{1}{N} \sum_{j=1}^{N} y_{A}^{(j)} \right)^{2}

- Saltelli2002: Requires :math:`N(2m + 2)` model evaluations :cite:`saltelli_2002`.

2. **Second order indices** (:math:`S_{ij}`)

- Saltelli2002: Requires :math:`N(2m + 2)` model evaluations :cite:`saltelli_2002`.

3. **Total order indices** (:math:`S_{T_{i}}`)

- Homma1996: Requires :math:`N(m + 1)` model evaluations :cite:`saltelli_2008`.

.. math:: 
   S_{T_{i}} = 1 - \frac{\mathbb{V}\left[E\left(Y \mid \mathbf{X}_{\sim_{i}}\right)\right]}{\mathbb{V}(Y)} = 1 - \frac{ (1 / N) Y_{B} \cdot Y_{C_{i}}-f_{0}^{2}}{(1 / N) Y_{A} \cdot Y_{A}-f_{0}^{2}}

.. math:: 
   y_{A}=f(A), \quad y_{B}=f(B), \quad y_{C_{i}}=f(C_{i}), \quad f_{0}^{2}=\left(\frac{1}{2N} \sum_{j=1}^{N} y_{A}^{(j)} + y_{B}^{(j)} \right)^{2}

- Saltelli2002: Requires :math:`N(2m + 2)` model evaluations :cite:`saltelli_2002`.


Sobol Class
^^^^^^^^^^^^^^^^^^

The :class:`.SobolSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.SobolSensitivity import SobolSensitivity


Methods
"""""""

.. autoclass:: UQpy.sensitivity.SobolSensitivity
     :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.first_order_indices
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.second_order_indices
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.total_order_indices
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.first_order_confidence_interval
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.second_order_confidence_interval
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.total_order_confidence_interval
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.n_samples
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.n_variables
.. autoattribute:: UQpy.sensitivity.SobolSensitivity.is_multi_output


Examples
""""""""""

.. toctree::

   Sobol Examples <../auto_examples/sensitivity/sobol/index>
