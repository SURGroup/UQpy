BayesParameterEstimation
-------------------------

Given some data :math:`\mathcal{D}`, a parameterized model for the data, and a prior probability density for the model
parameters :math:`p(\theta)`, the :class:`.BayesParameterEstimation` class is leveraged to draw samples from the posterior
pdf of the model parameters using Markov Chain Monte Carlo or Importance Sampling. Via Bayes theorem, the posterior pdf
is defined as follows:

.. math:: p(\theta \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert \theta)p(\theta)}{p(\mathcal{D})}

Note that if no prior is defined in the model, the prior pdf is chosen as uninformative, i.e., :math:`p(\theta) = 1` (cautionary note, this is an improper prior).

The :class:`.BayesParameterEstimation` leverages the :class:`.MCMC` or :class:`ImportanceSampling` classes of the
:py:mod:`.sampling` module of :py:mod:`UQpy`. When creating a :class:`.BayesParameterEstimation` object, an object of
class :class:`.MCMC` or :class:`.ImportanceSampling` is created and saved as an attribute `sampler`. The :meth:`run`
method of the :meth:`.BayesParameterEstimation` class then calls the:meth:`run` method of that sampler, thus the user
can add samples as they wish by calling the :meth:`run` method several times.


BayesParameterEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.inference.BayesParameterEstimation
    :members: