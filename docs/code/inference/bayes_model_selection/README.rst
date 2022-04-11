Bayesian Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The problem of model selection consists in determining which model(s) best explain the available data :math:`D`,
given a set of candidate models :math:`m_{1:M}`. Each model :math:`m_{j}` is parameterized by a set of parameters
:math:`\theta_{m_{j}} \in \Theta_{m_{j}}`, to be estimated based on data. In the Bayesian framework, model selection is
performed by computing the posterior probability of each model :math:`m_{j}` using Bayes' theorem:

.. math:: P(m_{j} \vert D) = \frac{p(D \vert m_{j})P(m_{j})}{\sum_{j=1}^{M} P(D \vert m_{j})P(m_{j})}

where :math:`P(m_{j})` is the prior assigned to model :math:`m_{j}` and :math:`P(D \vert m_{j})` is the model evidence, also called marginal likelihood.

.. math:: p(D \vert m_{j}) = \int_{\Theta_{m_{j}}} p(D \vert m_{j}, \theta_{m_{j}}) p(\theta_{m_{j}} \vert m_{j}) d\theta_{m_{j}}

where :math:`p(\theta_{m_{j}} \vert m_{j})` is the prior assigned to the parameter vector of model :math:`m_{j}`.