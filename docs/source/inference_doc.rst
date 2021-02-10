.. _inference_doc:

Inference
=============

.. automodule:: UQpy.Inference

The goal in inference can be twofold: 1) given a model, parameterized by parameter vector :math:`\theta`, and some data :math:`\mathcal{D}`, learn the value of the parameter vector that best explains the data; 2) given a set of candidate models :math:`\lbrace m_{i} \rbrace_{i=1:M}` and some data :math:`\mathcal{D}`, learn which model best explains the data. ``UQpy`` currently supports the following inference algorithms for parameter estimation (see e.g. [1]_ for theory on parameter estimation in frequentist vs. Bayesian frameworks):

* Maximum Likelihood estimation,
* Bayesian approach: estimation of posterior pdf via sampling methods (MCMC/IS).

and the following algorithms for model selection:

* Model selection using information theoretic criteria,
* Bayesian model class selection, i.e., estimation of model posterior probabilities.

The capabilities of ``UQpy`` and associated classes are summarized in the following figure.

.. image:: _static/Inference_schematic.png
   :scale: 40 %
   :align: left


InferenceModel
--------------------------------

For any inference task, the user must first create, for each model studied, an instance of the class ``InferenceModel`` that defines the problem at hand. This class defines an inference model that will serve as input for all remaining inference classes. A model can be defined in various ways. The following summarizes the four types of inference models that are supported by ``UQpy``. These four types are further summarized in the figure below.

* **Case 1a** - `Gaussian error model powered by` ``RunModel``: In this case, the data is assumed to come form a model of the following
  form,  `data ~ h(theta) + eps`, where `eps` is iid Gaussian and `h` consists of a computational
  model executed using ``RunModel``. Data is a 1D ndarray in this setting.
* **Case 1b** - `non-Gaussian error model powered by` ``RunModel``: In this case, the user must provide the likelihood
  function in addition to a ``RunModel`` object. The data type is user-defined and must be consistent with the
  likelihood function definition.
* **Case 2:** - `User-defined likelihood without` ``RunModel``: Here, the likelihood function is user-defined and
  does not leverage ``RunModel``. The data type must be consistent with the likelihood function definition.
* **Case 3:** `Learn parameters of a probability distribution:` Here, the user must define an object of the
  ``Distribution`` class. Data is an ndarray of shape `(ndata, dim)` and consists in `ndata` iid samples from the
  probability distribution.
	  
.. image:: _static/Inference_models.png
   :scale: 30 %
   :align: left
   

Defining a Log-likelihood function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The critical component of the ``InferenceModel`` class is the evaluation of the log-likelihood function. ``InferenceModel`` has been constructed to be flexible in how the user specifies the log-likelihood function. The log-likelihood function can be specified as a user-defined callable method that is passed directly into the ``InferenceModel`` class. As the cases suggest, a user-defined log-likelihood function must take as input, at minimum, both the parameters of the model and the data points at which to evaluate the log-likelihood. It may also take additional keyword arguments. The method may compute the log-likelihood at the data points on its own, or it may rely on a computational model defined through the ``RunModel`` class. If the log-likelihood function relies on a ``RunModel`` object, this object is also passed into ``InferenceModel`` and the log-likelihood method should also take as input, the output (`qoi_list`) of the ``RunModel`` object evaluated at the specified parameter values. 
   
InferenceModel Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
.. autoclass:: UQpy.Inference.InferenceModel
   :members:
   
Parameter estimation
--------------------------------

Parameter estimation refers to process of estimating the parameter vector of a given model. Depending on the nature of the method, parameter estimation may provide a point estimator or a probability distribution for the parameter vector. ``UQpy`` supports two different types of parameter estimation: Maximum Likelihood estimation through the ``MLEstimation`` class and Bayesian parameter estimation through the ``BayesParameterEstimation`` class.

MLEstimation
--------------

The ``MLEstimation`` class evaluates the maximum likelihood estimate :math:`\hat{\theta}` of the model parameters, i.e.

.. math:: \hat{\theta} = \text{argmax}_{\Theta} \quad p(\mathcal{D} \vert \theta)

Note: for a Gaussian-error model of the form :math:`\mathcal{D}=h(\theta)+\epsilon`, :math:`\epsilon \sim N(0, \sigma)` with fixed :math:`\sigma` and independent measurements :math:`\mathcal{D}_{i}`, maximizing the likelihood is mathematically equivalent to minimizing the sum of squared residuals :math:`\sum_{i} \left( \mathcal{D}_{i}-h(\theta) \right)^{2}`.

A numerical optimization procedure is performed to compute the MLE. By default, the `minimize` function of the ``scipy.optimize`` module is used, however other optimizers can be leveraged via the `optimizer` input of the  ``MLEstimation`` class.

MLEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Inference.MLEstimation
   :members:
   
**Note on subclassing** ``MLEstimation``

More generally, the user may want to compute a parameter estimate by minimizing an error function between the data and model outputs. This can be easily done by subclassing the ``MLEstimation`` class and overwriting the method `_evaluate_func_to_minimize`.

	
BayesParameterEstimation
-------------------------

Given some data :math:`\mathcal{D}`, a parameterized model for the data, and a prior probability density for the model parameters :math:`p(\theta)`, the ``BayesParameterEstimation`` class is leveraged to draw samples from the posterior pdf of the model parameters using Markov Chain Monte Carlo or Importance Sampling. Via Bayes theorem, the posterior pdf is defined as follows:

.. math:: p(\theta \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert \theta)p(\theta)}{p(\mathcal{D})}

Note that if no prior is defined in the model, the prior pdf is chosen as uninformative, i.e., :math:`p(\theta) = 1` (cautionary note, this is an improper prior).

The ``BayesParameterEstimation`` leverages the ``MCMC`` or ``IS`` classes of the ``SampleMethods`` module of ``UQpy``. When creating a ``BayesParameterEstimation`` object, an object of class ``MCMC`` or ``IS`` is created and saved as an attribute `sampler`. The ``run`` method of the ``BayesParameterEstimation`` class then calls the ``run`` method of that sampler, thus the user can add samples as they wish by calling the ``run`` method several times.


BayesParameterEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Inference.BayesParameterEstimation
    :members: 

Model Selection
----------------------------------

Model selection refers to the task of selecting a statistical model from a set of candidate models, given some data. A good model is one that is capable of explaining the data well. Given models of the same explanatory power, the simplest model should be chosen (Occam's razor). 

InfoModelSelection
--------------------

The ``InfoModelSelection`` class employs information-theoretic criteria for model selection. Several simple information theoretic criteria can be used to compute a model's quality and perform model selection [2]_. ``UQpy`` implements three criteria: 

* Bayesian information criterion,  :math:`BIC = \ln(n) k - 2 \ln(\hat{L})`
* Akaike information criterion, :math:`AIC = 2 k - 2 \ln (\hat{L})`
* Corrected formula for AIC (AICc), for small data sets , :math:`AICc = AIC + \frac{2k(k+1)}{n-k-1}`

where :math:`k` is the number of parameters characterizing the model, :math:`\hat{L}` is the maximum value of the likelihood function, and :math:`n` is the number of data points. The best model is the one that minimizes the criterion, which is a combination of a model fit term (find the model that minimizes the negative log likelihood) and a penalty term that increases as the number of model parameters (model complexity) increases. 

A probability can be defined for each model as :math:`P(m_{i}) \propto \exp\left(  -\frac{\text{criterion}}{2} \right)`.

InfoModelSelection Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Inference.InfoModelSelection
    :members: 
	
BayesModelSelection
---------------------

In the Bayesian approach to model selection, the posterior probability of each model is computed as

.. math:: P(m_{i} \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert m_{i})P(m_{i})}{\sum_{j} p(\mathcal{D} \vert m_{j})P(m_{j})}

where the evidence (also called marginal likelihood) :math:`p(\mathcal{D} \vert m_{i})` involves an integration over the parameter space:

.. math:: p(\mathcal{D} \vert m_{i}) = \int_{\Theta} p(\mathcal{D} \vert m_{i}, \theta) p(\theta \vert m_{i}) d\theta

Currently, calculation of the evidence is performed using the method of the harmonic mean [3]_:

.. math:: p(\mathcal{D} \vert m_{i}) = \left[ \frac{1}{B} \sum_{b=1}^{B} \frac{1}{p(\mathcal{D} \vert m_{i}, \theta_{b})} \right]^{-1}

where :math:`\theta_{1,\cdots,B}` are samples from the posterior pdf of :math:`\theta`. In UQpy, these samples are obtained via the ``BayesParameterEstimation`` class. However, note that this method is known to yield evidence estimates with large variance. Future releases of ``UQpy`` will include more robust methods for computation of model evidences. Also, it is known that results of such Bayesian model selection procedure usually highly depends on the choice of prior for the parameters of the competing models, thus the user should carefully define such priors when creating instances of the ``InferenceModel`` class.

BayesModelSelection Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Inference.BayesModelSelection
    :members: 

.. [1] R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014
.. [2] Burnham, K. P. and Anderson, D. R., "Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach", Springer-Verlag, 2002
.. [3] A.E. Raftery, M.A. Newton, J.M. Satagopan and P.N. Krivitsky, "Estimating the Integrated Likelihood via Posterior Simulation Using the Harmonic Mean Identity", Bayesian Statistics 8, 2007

.. toctree::
    :maxdepth: 2



	
	