Model selection using information criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Μodel selection refers to the task of selecting a statistical model from a set of candidate models,
given some data. A good model is a model that is able to explain the data well (high model evidence). Given models of
same explanatory power, the simplest model should be chosen (Ockam razor). Several simple criteria can be used to
compute a model's quality and thus perform model selection. UQpy implements three criteria:

Bayesian information criterion (BIC)

.. math:: BIC = ln(n) k - 2 ln(\hat{L})

Akaike information criterion (AIC)\

.. math:: AIC = 2 k - 2 ln (\hat{L})

Corrected formula for AIC (AICc), for small data sets

.. math:: AICc = AIC + \frac{2k(k+1)}{n-k-1}

For all formula above, :math:`k` is the number of parameters characterizing the model, :math:`\hat{L}` is the maximum value of the
likelihood function and :math:`n` the number of data points. The best model is the one that minimizes the criterion. All
three formulas have a model fit term (find the model that minimizes the negative log likelihood) and a penalty term
that increases as the number of model parameters (model complexity) increases.

Reference: *Burnham, K. P.; Anderson, D. R. (2002), Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach (2nd ed.), Springer-Verlag, ISBN 0-387-95364-7*