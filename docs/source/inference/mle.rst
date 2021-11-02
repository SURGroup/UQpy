MLEstimation
--------------

The :class:`.MLE` class evaluates the maximum likelihood estimate :math:`\hat{\theta}` of the model parameters, i.e.

.. math:: \hat{\theta} = \text{argmax}_{\Theta} \quad p(\mathcal{D} \vert \theta)

Note: for a Gaussian-error model of the form :math:`\mathcal{D}=h(\theta)+\epsilon`, :math:`\epsilon \sim N(0, \sigma)` with fixed :math:`\sigma` and independent measurements :math:`\mathcal{D}_{i}`, maximizing the likelihood is mathematically equivalent to minimizing the sum of squared residuals :math:`\sum_{i} \left( \mathcal{D}_{i}-h(\theta) \right)^{2}`.

A numerical optimization procedure is performed to compute the MLE. By default, the `minimize` function of the
:py:mod:`scipy.optimize`` module is used, however other optimizers can be leveraged via the `optimizer` input of the
:class:`.MLE` class.

MLEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.inference.MLE
   :members:
   :private-members:

.. Note::

   **Note on subclassing** :class:`MLE`

   More generally, the user may want to compute a parameter estimate by minimizing an error function between the data
   and model outputs. This can be easily done by subclassing the :class:`MLE` class and overwriting the method
   `_evaluate_func_to_minimize`.
