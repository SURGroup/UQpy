.. _surrogates_doc:


Surrogates
==========

.. automodule:: UQpy.Surrogates


Stochatic Reduced Order Models - SROMs
----------------------------------------

An SROM is a sample-based surrogate for probability models. An SROM takes a set of samples and attributes of a distribution and optimizes the sample probability weights according to the method in [1]_. More specifically, an SROM constructs a reduce order model for arbitrary random variables `X` as follows.

.. math:: \tilde{X} =  \begin{cases} x_1 & probability \text{  }p_1^{(opt)} \\ & \vdots \\ x_m & probability \text{  }p_m^{(opt)} \end{cases}

where :math:`\tilde{X}` is defined by an arbitrary set of samples :math:`x_1, \dots, x_m` defined over the same support as :math:`X` (but not necessarily drawn from its probability distribution) and their assigned probability weights. The probability weights are defined such that the total error between the sample empirical probability distribution, moments and correlation of :math:`\tilde{X}` and those of the random variable `X` is minimized. This optimization problem can be express as:

.. math:: & \min_{\mathbf{p}}  \sum_{u=1}^3 \alpha_u e_u(\mathbf{p}) \\ & \text{s.t.} \sum_{k=1}^m p_k =1 \quad and \quad p_k \geq 0, \quad k=1,2,\dots,m

where :math:`\alpha_1`, :math:`\alpha_2`, :math:`\alpha_3 \geq 0` are constants defining the relative importance of the marginal distribution, moments and correlation error between the reduce order model and actual random variables in the objective function.

.. math:: &  e_{1}(p)=\sum\limits_{i=1}^d \sum\limits_{k=1}^m w_{F}(x_{k,i};i)(\hat{F}_{i}(x_{k,i})-F_{i}(x_{k,i}))^2  \\ & e_{2}(p)=\sum\limits_{i=1}^d \sum\limits_{r=1}^2 w_{\mu}(r;i)(\hat{\mu}(r;i)-\mu(r;i))^2 \\ & e_{3}(p)=\sum\limits_{i,j=1,...,d ; j>i}  w_{R}(i,j)(\hat{R}(i,j)-R(i,j))^2

Here, :math:`F(x_{k,i})` and :math:`\hat{F}(x_{k,i})` denote the marginal cumulative distributions of :math:`\mathbf{X}` and :math:`\mathbf{\tilde{X}}` (reduced order model) evaluated at point :math:`x_{k,i}`, :math:`\mu(r; i)` and :math:`\hat{\mu}(r; i)` are the marginal moments of order `r` for variable `i`, and :math:`R(i,j)` and :math:`\hat{R}(i,j)` are correlation matrices of :math:`\mathbf{X}` and :math:`\mathbf{\tilde{X}}` evaluted for components :math:`x_i` and :math:`x_j`. Note also that `m` is the number of sample points and `d` is the number of random variables. 

SROM Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.Surrogates.SROM
    :members:


Gaussian Process Regression / Kriging
---------------------------------------

The ``Kriging`` class defines an approximate surrogate model or response surface which can be used to predict the model response and its uncertainty at points wehre the model has not been previously evaluated. Kriging gives the best unbiased linear predictor at the interpolated points. This class generates a model :math:`\hat{y}` that express the response as a realization of regression model and Gaussian random process as:

.. math:: \hat{y}(x) = \mathcal{F}(\beta, x) + z(x).

The regression model (:math:`\mathcal{F}`) is given as a linear combination of ':math:`p`' chosen scalar basis functions as:

.. math:: \mathcal{F}(\beta, x) = \beta_1 f_1(x) + \dots + \beta_p f_p(x) = f(x)^T \beta.

The random process :math:`z(x)` has zero mean and its covariance is defined through the separable correlation function:

.. math:: E\big[z(s)z(x)] = \sigma^2 \mathcal{R}(\theta, s, x)

where,

.. math:: \mathcal{R}(s, x; \theta) = \prod_{i=1}^d \mathcal{R}_i(s_i, x_i; \theta_i),

and :math:`\theta` are a set of hyperparameters generally governing the correlation length of the model determined by maximixing the log-likelihood function 

.. math:: \text{log}(p(y|x, \theta)) = -\frac{1}{2}y^T \mathcal{R}^{-1} y - \frac{1}{2}\text{log}(|\mathcal{R}|) - \frac{n}{2}\text{log}(2\pi)


The correlation is evaluated between a set of existing sample points :math:`s` and points :math:`x` in the domain of interest to form the correlation matrix :math:`R`, and the basis functions are evaluated at the sample points :math:`s` to form the matrix :math:`F`. Using these matrices, the regression coefficients, :math:`\beta`, and process variance, :math:`\sigma^2` are computed as

.. math:: (F^T R^{-1} F)\beta^* & = F^T R^{-1} Y \\ \sigma^2 & = \frac{1}{m} (Y - F\beta^*)^T R{-1}(Y - F\beta^*)

The final predictor function is then given by:

.. math:: \hat{y}(x) = f(x)^T \beta^* + r(x)^T R^{-1}(Y - F\beta^*)

Regression Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Kriging`` class offers a variety of built-in regression models, specified by the `reg_model` input described below.


Ordinary Kriging
~~~~~~~~~~~~~~~~~~~~

In ordinary Kriging, the regression model is assumed to take a constant value such that 

.. math:: \mathcal{F}(\beta, x) = \beta_0


Universal Kriging
~~~~~~~~~~~~~~~~~~~~~

In universal Kriging, the regression model is assumed to take a general functional form. The ``Kriging`` class currenly supports two univeral Kriging models, the linear regression model given by:

.. math:: \mathcal{F}(\beta, x) = \beta_0 = \sum_{i=1}^d \beta_i x_i

and the quadratic regression model given by:

.. math:: \mathcal{F}(\beta, x) = \beta_0 = \sum_{i=1}^d \beta_i x_i + \sum_{i=1}^d \sum_{j=1}^d \beta_{ij} x_i x_j


User-Defined Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new regression model to the ``Kriging`` class is straightforward. This is done by creating a new method that evaluates the basis functions and the Jacobian. This method may be passed directly as a callable to the `reg_model` input of the ``Kriging`` class. This new method should takes as input the samples points at which to evaluate the model and return two arrays containing the value of the basis functions and the Jacobian at these sample points. 

The first output of this function should be a two dimensional numpy array with the first dimension being the number of samples and the second dimension being the number of basis functions. 

The second output (i.e. Jacobian of basis function) is a three dimensional numpy array with the first dimension being the number of samples, the second dimension being the number of variables and the third dimension being the number of basis functions. 

An example user-defined model is given below:


>>> def constant(points):
>>> 	fx = np.ones([points.shape[0], 1])
>>> 	jf = np.zeros([points.shape[0], points.shape[1], 1])
>>> 	return fx, jf

Correlation Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Kriging`` class offers a variety of built-in correlation models, specified by the `corr_model` input described below. 

Exponential Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The exponential correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = \exp\bigg[ -\dfrac{|h_i|}{\theta_i}\bigg]

where :math:`h_i = s_i-x_i`.


Gaussian Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gaussian correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = \exp\bigg[ -\bigg(\dfrac{h_i}{\theta_i}\bigg)^2\bigg]

where :math:`h_i = s_i-x_i`.

Linear Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = \max \bigg(0, 1-\dfrac{|h_i|}{\theta_i}\bigg)

where :math:`h_i = s_i-x_i`.


Spherical Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The spherical correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = 1 - 1.5\xi_i + 0.5\xi_i^3

where :math:`\xi_i =  \min \bigg(1, \dfrac{|h_i|}{\theta_i}\bigg)` and :math:`h_i = s_i-x_i`.



Cubic Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The cubic correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = 1 - 3\xi_i^2 + 2\xi_i^3

where :math:`\xi_i =  \min \bigg(1, \dfrac{|h_i|}{\theta_i}\bigg)` and :math:`h_i = s_i-x_i`.


Spline Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The spline correlation model takes the following form:

.. math:: \mathcal{R}_i(h_i, \theta_i) = \begin{cases} 1-1.5\xi_i^2+30\xi_i^3, & 0\leq \xi_i \leq 0.02 \\  1.25(1-\xi_i)^3, & 0.2<\xi_i<1 \\ 0, & \xi_i \geq 1\end{cases}

where :math:`\xi_i =  \min \bigg(1, \dfrac{|h_i|}{\theta_i}\bigg)` and :math:`h_i = s_i-x_i`.


User-Defined Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~


Adding a new correlation model to the ``Kriging`` class is straightforward. This is done by creating a new method that evaluates the correlation matrix, it's derivative with respect to the variables and it's derivative with respect to the hyperparameters. This method takes as input the new points, training points, hyperparameters and two indicators for the computation of the derivative of correlation matrix (i.e. `dt` and `dx`). 

If both indicators are false, then the method should return correlation matrix, i.e. a 2-D array with first dimension being the number of points and second dimension being the number of training points. 

If `dx` parameter is True, the method should return the derivative of the correlation matrix respect to the variables, i.e. a 3-D array with first dimension being the number of points, second dimension being the number of training points and third dimension being the number of variables. 

If `dt` is True, then the method should return the correlation matrix and it's derivative with respect to the hyperparameters, i.e. a 3-D array with first dimension being the number of points, second dimension being the number of training points and third dimension being the number of variables. 

An example user-defined model is given below:


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

Kriging Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Surrogates.Kriging
	:members:

.. [1] M. Grigoriu, “Reduced order models for random functions. Application to stochastic problems”, Applied Mathematical Modelling, Volume 33, Issue 1, Pages 161-175, 2009.

.. toctree::
    :maxdepth: 2



	
	