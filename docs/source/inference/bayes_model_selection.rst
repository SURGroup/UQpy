BayesModelSelection
---------------------

In the Bayesian approach to model selection, the posterior probability of each model is computed as

.. math:: P(m_{i} \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert m_{i})P(m_{i})}{\sum_{j} p(\mathcal{D} \vert m_{j})P(m_{j})}

where the evidence (also called marginal likelihood) :math:`p(\mathcal{D} \vert m_{i})` involves an integration over the parameter space:

.. math:: p(\mathcal{D} \vert m_{i}) = \int_{\Theta} p(\mathcal{D} \vert m_{i}, \theta) p(\theta \vert m_{i}) d\theta

Currently, calculation of the evidence is performed using the method of the harmonic mean :cite:`BayesModelSelection`:

.. math:: p(\mathcal{D} \vert m_{i}) = \left[ \frac{1}{B} \sum_{b=1}^{B} \frac{1}{p(\mathcal{D} \vert m_{i}, \theta_{b})} \right]^{-1}

where :math:`\theta_{1,\cdots,B}` are samples from the posterior pdf of :math:`\theta`. In UQpy, these samples are
obtained via the :class:`.BayesParameterEstimation` class. However, note that this method is known to yield evidence
estimates with large variance. The Harmonic mean method for calculating the evidence is provided to the initializer
of the :class:`.BayesModelSelection` class. No input parameters are required for its initialization.

.. autoclass:: UQpy.inference.HarmonicMean


Future releases of :py:mod:`UQpy` will include more robust methods for computation of model
evidences. Also, it is known that results of such Bayesian model selection procedure usually highly depends on the
choice of prior for the parameters of the competing models, thus the user should carefully define such priors when
creating instances of the :class:`.InferenceModel` class. The user can create custom methods for calculating
evidences, by extending the :class:`.EvidenceMethod` abstract baseclass. To achieve that, a custom implementation
of the :meth:`.estimate_evidence` must be provided.


EvidenceMethod Class
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.inference.EvidenceMethod
    :members:



BayesModelSelection Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.inference.BayesModelSelection
    :members: run, sort_models

Attributes
""""""""""
.. autoattribute:: UQpy.inference.BayesModelSelection.candidate_models
.. autoattribute:: UQpy.inference.BayesModelSelection.bayes_estimators
.. autoattribute:: UQpy.inference.BayesModelSelection.evidences
.. autoattribute:: UQpy.inference.BayesModelSelection.probabilities