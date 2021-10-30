
Inference
=============

.. automodule:: UQpy.inference

The goal in inference can be twofold: 1) given a model, parameterized by parameter vector :math:`\theta`, and some data :math:`\mathcal{D}`, learn the value of the parameter vector that best explains the data; 2) given a set of candidate models :math:`\lbrace m_{i} \rbrace_{i=1:M}` and some data :math:`\mathcal{D}`, learn which model best explains the data. ``UQpy`` currently supports the following inference algorithms for parameter estimation (see e.g. [1]_ for theory on parameter estimation in frequentist vs. Bayesian frameworks):

* Maximum Likelihood estimation,
* Bayesian approach: estimation of posterior pdf via sampling methods (MCMC/IS).

and the following algorithms for model selection:

* Model selection using information theoretic criteria,
* Bayesian model class selection, i.e., estimation of model posterior probabilities.

The capabilities of ``UQpy`` and associated classes are summarized in the following figure.


.. image:: ../_static/Inference_schematic.png
   :scale: 40 %
   :align: left


.. toctree::
   :maxdepth: 2
   :caption: Inference

    Inference Models <inference_models>
    Maximum Likelihood Estimation <mle>
    Bayes Parameter Estimation <bayes_parameter_estimation>
    Information Theoretic Model Selection <info_model_selection>
    Bayes Model Selection <bayes_parameter_estimation>


.. [1] R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014
.. [2] Burnham, K. P. and Anderson, D. R., "Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach", Springer-Verlag, 2002
.. [3] A.E. Raftery, M.A. Newton, J.M. Satagopan and P.N. Krivitsky, "Estimating the Integrated Likelihood via Posterior Simulation Using the Harmonic Mean Identity", Bayesian Statistics 8, 2007
