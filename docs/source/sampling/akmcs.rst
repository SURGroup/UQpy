AdaptiveKriging
---------------

The :class:`.AdaptiveKriging` class generates samples adaptively using a specified Kriging-based learning function in a
general Adaptive Kriging-Monte Carlo Sampling (AKMCS) framework. Based on the specified learning function, different
objectives can be achieved. In particular, the :class:`.AdaptiveKriging` class has learning functions for reliability analysis
(probability of failure estimation), global optimization, best global fit surrogate models, and can also accept
user-defined learning functions for these and other objectives.  Note that the term AKMCS is adopted from [3]_ although
the procedure is referred to by different names depending on the specific learning function employed. For example,
when applied for optimization the algorithm leverages the expected improvement function and is known under the name
Efficient Global Optimization (EGO) [4]_.


Learning Functions
^^^^^^^^^^^^^^^^^^^^
:class:`.AdaptiveKriging` provides a number of built-in learning functions as well as allowing the user to provide a
custom learning function. These learning functions are described below.


U-Function
~~~~~~~~~~~~

The U-function is a learning function adopted for Kriging-based reliability analysis adopted from [3]_. Given a Kriging model :math:`\hat{y}(\mathbf{x})`, point estimator of its standard devaition :math:`\sigma_{\hat{y}}(\mathbf{x})`, and a set of learning points :math:`S`, the U-function seeks out the point :math:`\mathbf{x}\in S` that minimizes the function:

.. math:: U(\mathbf{x}) = \dfrac{|\hat{y}(\mathbf{x})|}{\sigma_{\hat{y}}(\mathbf{x})}

This point can be interpreted as the point in :math:`S` where the Kriging model has the highest probabability of incorrectly identifying the sign of the performance function (i.e. incorrectly predicting the safe/fail state of the system).

The :class:`.AdaptiveKriging` then adds the corresponding point to the training set, re-fits the Kriging model and repeats the procedure until the following stopping criterion in met:

.. math:: \min(U(\mathbf{x})) > \epsilon_u

where :math:`\epsilon_u` is a user-defined error threshold (typically set to 2).


Weighted U-Function
~~~~~~~~~~~~~~~~~~~~~

The probability weighted U-function is a learning function for reliability analysis adapted from the U-function in [5]_. It modifies the U-function as follows:

.. math:: W(\mathbf{x}) = \dfrac{\max_x[p(\mathbf{x})] - p(\mathbf{x})}{\max_x[p(\mathbf{x})]} U(\mathbf{x})

where :math:`p(\mathbf{x})` is the probability density function of :math:`\mathbf{x}`. This has the effect of decreasing the learning function for points that have higher probability of occurrence. Thus, given two points with identical values of :math:`U(x)`, the weighted learning function will select the point with higher probability of occurrence.

As with the standard U-function, :class:`.AdaptiveKriging` with the weighted U-function iterates until :math:`\min(U(\mathbf{x})) > \epsilon_u` (the same stopping criterion as the U-function).


Expected Feasibility Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Expected Feasibility Function (EFF) is a learning function for reliability analysis introduced as part of the Efficient Global Reliability Analysis (EGRA) method [6]_. The EFF provides assesses how well the true value of the peformance function, :math:`y(\mathbf{x})`, is expected to satisfy the constraint :math:`y(\mathbf{x}) = a` over a region :math:`a-\epsilon \le y(\mathbf{x}) \le a+\epsilon`. It is given by:

.. math:: \begin{align} EFF(\mathbf{x}) &= (\hat{y}(\mathbf{x})-a)\bigg[2\Phi\bigg(\dfrac{a-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) - \Phi\bigg(\dfrac{(a-\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) - \Phi\bigg(\dfrac{(a+\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) \bigg] \\ &-\sigma_{\hat{y}}(\mathbf{x})\bigg[2\phi\bigg(\dfrac{a-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) - \phi\bigg(\dfrac{(a-\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) - \phi\bigg(\dfrac{(a+\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) \bigg] \\ &+ \bigg[ \Phi\bigg(\dfrac{(a+\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) - \Phi\bigg(\dfrac{(a-\epsilon)-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) \bigg] \end{align}

where :math:`\Phi(\cdot)` and :math:`\phi(\cdot)` are the standard normal cdf and pdf, respectively. For reliabilty, :math:`a=0`, and it is suggest to use :math:`\epsilon=2\sigma_{\hat{y}}^2`.

At each iteration, the new point that is selected is the point that maximizes the EFF and iterations continue until

.. math:: \max_x(EFF(\mathbf{x})) < \epsilon_{eff}


Expected Improvement Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Expected Improvement Function (EIF) is a Kriging-based learning function for global optimization introduced as part of the Efficient Global Optimization (EGO) method in [4]_. The EIF seeks to find the global minimum of a function. It searches the space by placing samples at locations that maximize the expected improvement, where the improvement is defined as :math:`I(\mathbf{x})=\max(y_{min}-y(\mathbf{x}), 0)`, where the model response :math:`y(\mathbf{x})` is assumed to be a Gaussian random variable and :math:`y_{min}` is the current minimum model response. The EIF is then expressed as:

.. math:: EIF(\mathbf{x}) = E[I(\mathbf{x})] = (y_{min}-\hat{y}(\mathbf{x})) \Phi \bigg(\dfrac{y_{min}-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg) + \sigma_{\hat{y}}(\mathbf{x})\phi \bigg(\dfrac{y_{min}-\hat{y}(\mathbf{x})}{\sigma_{\hat{y}}(\mathbf{x})} \bigg)

where :math:`\Phi(\cdot)` and :math:`\phi(\cdot)` are the standard normal cdf and pdf, respectively.

At each iteration, the EGO algorithm selects the point in the learning set that maximizes the EIF. The algorithm continues until the maximum number of iterations or until:

.. math:: \dfrac{EIF(\mathbf{x})}{|y_{min}|} < \epsilon_{eif}.

Typically a value of 0.01 is used for :math:`\epsilon_{eif}`.


Expected Improvement for Global Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Expected Improvement for Global Fit (EIGF) learning function aims to build the surrogate model that is the best global representation of model. It was introduced in [7]_. It aims to balance between even space-filling design and sampling in regions of high variation and is given by:

.. math:: EIGF(\mathbf{x}) = (\hat{y}(\mathbf{x}) - y(\mathbf{x}_*))^2 + \sigma_{\hat{y}}(\mathbf{x})^2

where :math:`\mathbf{x}_*` is the point in the training set closest in distance to the point :math:`\mathbf{x}` and :math:`y(\mathbf{x}_*)` is the model response at that point.

No stopping criterion is suggested by the authors of [7]_, thus its implementation in :class:`.AdaptiveKriging` uses a fixed number of iterations.


User-Defined Learning Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.AdaptiveKriging` class also allows new, user-defined learning functions to be specified in a straightforward way. This is done by creating a new method that contains the algorithm for selecting a new samples. This method takes as input the surrogate model, the randomly generated learning points, the number of points to be added in each iteration, any requisite parameters including a stopping criterion, existing samples, model evaluate at samples and distribution object. It returns a set of samples that are selected according to the user's desired learning function and the corresponding learning function values. The outputs of this function should be (1) a numpy array of samples to be added; (2) the learning function values at the new sample points, and (3) a boolean stopping criterion indicating whether the iterations should continue (`False`) or stop (`True`). The numpy array of samples should be a two-dimensional array with the first dimension being the number of samples and the second dimension being the number of variables. An example user-defined learning function is given below:


>>> class UserLearningFunction(LearningFunction):
>>>
>>>    def __init__(self, u_stop: int = 2):
>>>        self.u_stop = u_stop
>>>
>>>    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
>>>        # AKMS class use these inputs to compute the learning function
>>>
>>>        g, sig = surrogate.predict(population, True)
>>>
>>>        # Remove the inconsistency in the shape of 'g' and 'sig' array
>>>        g = g.reshape([population.shape[0], 1])
>>>        sig = sig.reshape([population.shape[0], 1])
>>>
>>>        u = abs(g) / sig
>>>        rows = u[:, 0].argsort()[:n_add]
>>>
>>>        indicator = False
>>>        if min(u[:, 0]) >= self.u_stop:
>>>            indicator = True
>>>
>>>        return population[rows, :], u[rows, 0], indicator

AdaptiveKriging Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.sampling.AdaptiveKriging
    :members:


.. [3] B. Echard, N. Gayton and M. Lemaire, "AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation", Structural Safety, Pages 145-154, 2011.
.. [4] Jones, D. R., Schonlau, M., & Welch, W. J. "Efficient global optimization of expensive black-box functions." Journal of Global optimization, 13(4), 455-492, 1998.
.. [5] V.S. Sundar and Shields, M.D. "Reliablity analysis using adaptive Kriging surrogates and multimodel inference." ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems. Part A: Civil Engineering. 5(2): 04019004, 2019.
.. [6] B.J. Bichon, M.S. Eldred, L.P. Swiler, S. Mahadevan, and J.M. McFarland. "Efficient global reliablity analysis for nonlinear implicit performance functions." AIAA Journal. 46(10) 2459-2468, (2008).
.. [7] C.Q. Lam. "Sequential adaptive designs in computer experiments for response surface model fit." PhD diss., The Ohio State University, 2008.