InfoModelSelection
--------------------

The :class:`InformationModelSelection` class employs information-theoretic criteria for model selection. Several simple information
theoretic criteria can be used to compute a model's quality and perform model selection :cite:`InfoModelSelection`. :py:mod:`UQpy` implements three criteria:

* Bayesian information criterion,  :math:`BIC = \ln(n) k - 2 \ln(\hat{L})`

The :class:`.BIC` class is imported using the following command:

>>> from UQpy.inference.information_criteria.BIC import BIC

.. autoclass:: UQpy.inference.information_criteria.BIC

* Akaike information criterion, :math:`AIC = 2 k - 2 \ln (\hat{L})`

The :class:`.AIC` class is imported using the following command:

>>> from UQpy.inference.information_criteria.AIC import AIC

.. autoclass:: UQpy.inference.information_criteria.AIC

* Corrected formula for :math:`AIC (AICc)`, for small data sets , :math:`AICc = AIC + \frac{2k(k+1)}{n-k-1}`

The :class:`.AICc` class is imported using the following command:

>>> from UQpy.inference.information_criteria.AICc import AICc

.. autoclass:: UQpy.inference.information_criteria.AICc

where :math:`k` is the number of parameters characterizing the model, :math:`\hat{L}` is the maximum value of the likelihood function, and :math:`n` is the number of data points. The best model is the one that minimizes the criterion, which is a combination of a model fit term (find the model that minimizes the negative log likelihood) and a penalty term that increases as the number of model parameters (model complexity) increases.

A probability can be defined for each model as :math:`P(m_{i}) \propto \exp\left(  -\frac{\text{criterion}}{2} \right)`.

Note that none of the above information theoretic criteria requires any input parameters from initialization and thus their instances can be created as follows:

>>> criterion = AIC()

All of these criteria are child classes of the :class:`.InformationCriterion` abstract baseclass. The user can create
new type of criteria by extending the :class:`.InformationCriterion` and providing an alternative implementation to the
:py:meth:`evaluate_criterion` method.

The :class:`.InformationCriterion` class is imported using the following command:

>>> from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion

.. autoclass:: UQpy.inference.information_criteria.baseclass.InformationCriterion
    :members:

InfoModelSelection Class
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.InformationModelSelection` class is imported using the following command:

>>> from UQpy.inference.InformationModelSelection import InformationModelSelection

Methods
"""""""
.. autoclass:: UQpy.inference.InformationModelSelection
    :members: run, sort_models

Attributes
""""""""""
.. autoattribute:: UQpy.inference.InformationModelSelection.ml_estimators
.. autoattribute:: UQpy.inference.InformationModelSelection.criterion_values
.. autoattribute:: UQpy.inference.InformationModelSelection.penalty_terms
.. autoattribute:: UQpy.inference.InformationModelSelection.probabilities