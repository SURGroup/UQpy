.. _surrogates:


Surrogates
==========

.. automodule:: UQpy.Surrogates


SROM
----

The ``SROM`` takes a set of samples and attributes of a distribution and optimizes the sample probability weights according to the method of Stochastic Reduced Order Models as defined by [1]_. SROM constructs a reduce order model for arbitrary random variables.

.. math:: \tilde{X} =  \begin{cases} x_1 & probability \text{  }p_1^{(opt)} \\ & \vdots \\ x_m & probability \text{  }p_m^{(opt)} \end{cases}

This class identify the probability/weights associated with sample, such that total error between distribution, moments and correlation of random variables is minimized. This optimization problem can be express as:

.. math:: & \min_{\mathbf{p}}  \sum_{u=1}^3 \alpha_u e_u(\mathbf{p}) \\ & \sum_{k=1}^m p_k =1 \quad and \quad p_k \geq 0, \quad k=1,2,\dots,m

where :math:`\alpha_1`, :math:`\alpha_2`, :math:`\alpha_3 \geq 0` are constants defining the relative important of distribution, moments and correlation error between the reduce order model and actual random variables in the objective function.

.. math:: &  e_{1}(p)=\sum\limits_{i=1}^d \sum\limits_{k=1}^m w_{F}(x_{k,i};i)(\hat{F}_{i}(x_{k,i})-F_{i}(x_{k,i}))^2  \\ & e_{2}(p)=\sum\limits_{i=1}^d \sum\limits_{r=1}^q w_{\mu}(r;i)(\hat{\mu}(r;i)-\mu(r;i))^2 \\ & e_{3}(p)=\sum\limits_{i,j=1,...,d ; j>i}  w_{r}(i,j)(\hat{r}(i,j)-r(i,j))^2

Here, :math:`F` and :math:`\hat{F}` denote the marginal distribution of :math:`\mathbf{X}` and :math:`\mathbf{\hat{X}}` (reduced order model). Similarly, :math:`\mu` and :math:`\hat{\mu}` are marginal moments and :math:`r` and :math:`\hat{r}` are correlation matrix of :math:`\mathbf{X}` and :math:`\mathbf{\hat{X}}`. This class only consider first and second order moment about origin, i.e. q=2. And, 'm' is number of samples and 'd' is number of random variables.

Class Descriptions
^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.Surrogates.SROM
    :members:


Kriging
-------

The ``Kriging`` class defines an approximate surrogate model or response surface which can be used to predict function values at unknown location. Kriging gives the best unbiased linear predictor at the intermediate samples. This class generates a model :math:`\hat{y}` that express the response surface as a realization of regression model and gaussian random process.

.. math:: \hat{y}(x) = \mathcal{F}(\beta, x) + z(x)

Regression model (:math:`\mathcal{F}`) is linear combination of ':math:`p`' chosen scalar basis function.

.. math:: \mathcal{F}(\beta, x) = \beta_1 f_1(x) + \dots + \beta_p f_p(x) = f(x)^T \beta

The random process :math:`z(x)` have mean zero and covariance is defined through correlation matrix(:math:`\mathcal{R}(\theta, s, x)`), which depends on hyperparameters(:math:`\theta`) and samples(:math:`s`).

.. math:: E\big[z(s)z(x)] = \sigma^2 \mathcal{R}(\theta, s, x)

Hyperparameters are estimate by maximizing the log-likehood function.

.. math:: \text{log}(p(y|x, \theta)) = -\frac{1}{2}y^T \mathcal{R}^{-1} y - \frac{1}{2}\text{log}(|\mathcal{R}|) - \frac{n}{2}\text{log}(2\pi)

Once hyperparameters are computed, correlation matrix(:math:`\mathcal{R}`) and basis functions are evaluated at sample points(:math:`F`). Then, correlation coefficient(:math:`\beta`) and process variance(:math:`\sigma^2`) can be computed using following equations.

.. math:: (F^T R^{-1} F)\beta^* & = F^T R^{-1} Y \\ \sigma^2 & = \frac{1}{m} (Y - F\beta^*)^T R{-1}(Y - F\beta^*)

The final predictor function can be defined as:

.. math:: \hat{y}(x) = f(x)^T \beta^* + r(x)^T R^{-1}(Y - F\beta^*)

Adding New Regression Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Kriging`` class offers a variety of model for fitting approximate surrogate model. These are specified by the `reg_model` parameter (i.e. 'constant', 'linear' and 'quadratic'). However, adding a new model is straightforward. This is done by creating a new method that computes basis function and it's jacobian. This method takes as input the samples points and returns two array containing the value of value of basis function and it's jacobian at sample points. The first output of this function should be a two dimensional numpy array with the first dimension being the number of samples and the second dimension being the number of basis functions. The second output (i.e. jacobian of basis function) is a three dimensional numpy array with the first dimension being the number of samples, the second dimension being the number of variables and the third dimension being the number of basis functions. An example user-defined model is given below:


>>> def constant(points):
>>> 	fx = np.ones([points.shape[0], 1])
>>> 	jf = np.zeros([points.shape[0], points.shape[1], 1])
>>> 	return fx, jf

Adding New Correlation Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Kriging`` class offers a variety of model to minimize the error in surrogate model. These are specified by the `corr_model` parameter (i.e. 'exponential', 'gaussian', 'linear', 'spherical', 'cubic' and 'spline'). However, user can also add a new model. This is done by creating a new method that computes correlation matrix, it's derivative w.r.t samples and it's derivative w.r.t hyperparameters. This method takes as input the new points, training points, hyperparameters and two indicator for the computation of derivative of correlation matrix (i.e. `dt` and `dx`). If both indicators are false, then method should return correlation matrix, i.e. a 2-D array with first dimension being the number of points and second dimension being the number of training points. If `dx` parameter is True, then method should return correlation matrix and derivative of correlation matrix w.r.t variables, i.e. a 3-D array with first being the number of points, second dimension being the number of training points and third dimension being the number of variables. If `dt` is True, then method should return correlation matrix and it's derivative w.r.t hyperparameters, i.e. a 3-D array with same shape as derivative of correlation matrix w.r.t. variables. An example user-defined model is given below:


>>> def Gaussian(x, s, params, dt=False, dx=False):
>>>     x, s = np.atleast_2d(x), np.atleast_2d(s)
>>>     # Create stack matrix, where each block is x_i with all s
>>>     stack = - np.tile(np.swapaxes(np.atleast_3d(x), 1, 2), (1, np.size(s, 0), 1)) + np.tile(s, (np.size(x, 0), 1, 1))
>>>     rx = np.exp(np.sum(-params * (stack ** 2), axis=2))
>>>     if dt:
>>>         drdt = -(stack ** 2) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
>>>         return rx, drdt
>>>     if dx:
>>>         drdx = 2 * params * stack * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
>>>         return rx, drdx
>>>     return rx

Class Descriptions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Surrogates.Kriging
	:members:

.. [1] M. Grigoriu, “Reduced order models for random functions. Application to stochastic problems”, Applied Mathematical Modelling, Volume 33, Issue 1, Pages 161-175, 2009.
.. [2] S.N. Lophaven , Hans Bruun Nielsen , J. Søndergaard, "DACE -- A MATLAB Kriging Toolbox", Informatics and Mathematical Modelling, Version 2.0, 2002.

.. toctree::
    :maxdepth: 2



	
	