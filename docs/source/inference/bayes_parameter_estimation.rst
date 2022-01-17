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
class :class:`.MCMC` or :class:`.ImportanceSampling` is created and saved as an attribute :py:attr:`.sampler`. The :meth:`run`
method of the :class:`.BayesParameterEstimation` class then calls the :py:meth:`run` method of that :py:attr:`.sampler`, thus the user
can add samples as they wish by calling the :meth:`run` method several times.


BayesParameterEstimation Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.inference.BayesParameterEstimation
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.inference.BayesParameterEstimation.sampler

----

Below, an example of the :class:`.BayesParameterEstimation` usage is provided. The goal is to learn the parameters of
a :class:`.Normal` distribution.

- Initially, a two distributions are created based on the prior belief on each one of the unknown parameters and are subsequently merged into  joint distribution.

- The second step is to define the model whose parameters we want to infer. In this case as already mentioned the goal is to learn two parameters of the :class:`.Normal` distribution, so a :class:`.Distribution` model is defined, where the number of parameters `n_parameters`, as well as the prior distributions is provided.

- Before initializing the final :class:`.BayesParameterEstimation` object, the user must provide a method to sample the posterior distribution of the candidate model. Here a child class of :class:`.MCMC` is chosen and specifically, :class:`.MetropolisHastings`. Apart from the various parameters required by the algorithm such as `jump` or `burn_length`, the user must specify the `args_target` and `log_pdf_target` parameters as follows:

    - ``args_target = (data, )``
    - ``log_pdf_target = candidate_model.evaluate_log_posterior``

- Finally, the :class:`BayesParameterEstimation` object is created, with input, the sampling, the candidate model whose parameters we want to learn, the data, as well as the number of samples to be drawn from the posterior distribution. Since `nsamples` is provided at the object initialization, the :class:`.BayesParameterEstimation` will be automatically performed. Alternatively, the user must call the :py:meth:`run` method.


>>> p0 = Uniform(loc=0., scale=15)
>>> p1 = Lognormal(s=1., loc=0., scale=1.)
>>> prior = JointIndependent(marginals=[p0, p1])
>>>
>>> # create an instance of class Model
>>> candidate_model = DistributionModel(distributions=Normal(loc=None, scale=None),
>>>                                     n_parameters=2, prior=prior)
>>>
>>> sampling = MetropolisHastings(jump=10, burn_length=10, seed=[1.0, 0.2], random_state=1,
>>>                               args_target=(data, ),
>>>                               log_pdf_target=candidate_model.evaluate_log_posterior)
>>>
>>> bayes_estimator = BayesParameterEstimation(sampling_class=sampling,
>>>                                            inference_model=candidate_model,
>>>                                            data=data,
>>>                                            nsamples=5)

