MLE
--------------

The :class:`.MLE` class evaluates the maximum likelihood estimate :math:`\hat{\theta}` of the model parameters, i.e.

.. math:: \hat{\theta} = \text{argmax}_{\Theta} \quad p(\mathcal{D} \vert \theta)

Note: for a Gaussian-error model of the form :math:`\mathcal{D}=h(\theta)+\epsilon`, :math:`\epsilon \sim N(0, \sigma)`
with fixed :math:`\sigma` and independent measurements :math:`\mathcal{D}_{i}`, maximizing the likelihood is
mathematically equivalent to minimizing the sum of squared residuals :math:`\sum_{i} \left( \mathcal{D}_{i}-h(\theta) \right)^{2}`.

A numerical optimization procedure is performed to compute the MLE. By default, the :py:meth:`minimize` function of the
:py:mod:`scipy.optimize` module is used, however other optimizers can be leveraged via the `optimizer` input of the
:class:`.MLE` class.

MLE Class
^^^^^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.inference.MLE
   :members: run


Attributes
""""""""""
.. autoattribute:: UQpy.inference.MLE.mle
.. autoattribute:: UQpy.inference.MLE.max_log_like