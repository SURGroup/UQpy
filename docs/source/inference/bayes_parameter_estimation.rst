BayesParameterEstimation
-------------------------

Given some data :math:`\mathcal{D}`, a parameterized model for the data, and a prior probability density for the model parameters :math:`p(\theta)`, the ``BayesParameterEstimation`` class is leveraged to draw samples from the posterior pdf of the model parameters using Markov Chain Monte Carlo or Importance Sampling. Via Bayes theorem, the posterior pdf is defined as follows:

.. math:: p(\theta \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert \theta)p(\theta)}{p(\mathcal{D})}

Note that if no prior is defined in the model, the prior pdf is chosen as uninformative, i.e., :math:`p(\theta) = 1` (cautionary note, this is an improper prior).

The ``BayesParameterEstimation`` leverages the ``MCMC`` or ``IS`` classes of the ``SampleMethods`` module of ``UQpy``. When creating a ``BayesParameterEstimation`` object, an object of class ``MCMC`` or ``IS`` is created and saved as an attribute `sampler`. The ``run`` method of the ``BayesParameterEstimation`` class then calls the ``run`` method of that sampler, thus the user can add samples as they wish by calling the ``run`` method several times.


BayesParameterEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.inference.BayesParameterEstimation
    :members: