InfoModelSelection
--------------------

The :class:`InformationModelSelection` class employs information-theoretic criteria for model selection. Several simple information
theoretic criteria can be used to compute a model's quality and perform model selection :cite:`InfoModelSelection`. :py:mod:`UQpy` implements three criteria:

* Bayesian information criterion,  :math:`BIC = \ln(n) k - 2 \ln(\hat{L})`
* Akaike information criterion, :math:`AIC = 2 k - 2 \ln (\hat{L})`
* Corrected formula for AIC (AICc), for small data sets , :math:`AICc = AIC + \frac{2k(k+1)}{n-k-1}`

where :math:`k` is the number of parameters characterizing the model, :math:`\hat{L}` is the maximum value of the likelihood function, and :math:`n` is the number of data points. The best model is the one that minimizes the criterion, which is a combination of a model fit term (find the model that minimizes the negative log likelihood) and a penalty term that increases as the number of model parameters (model complexity) increases.

A probability can be defined for each model as :math:`P(m_{i}) \propto \exp\left(  -\frac{\text{criterion}}{2} \right)`.

Depending on which criterion will be utilized, the user must import the :class:`.InformationTheoreticCriterion` as
follows:

>>> from UQpy.inference.InformationTheoreticCriterion import InformationTheoreticCriterion

The definition of the latter enumeration options is given below:

.. autoclass:: UQpy.inference.InformationTheoreticCriterion
    :members:

InfoModelSelection Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.inference.InformationModelSelection
    :members:
    :private-members:
