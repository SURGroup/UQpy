Gaussian Process Regression / Kriging
---------------------------------------

The :class:`.Kriging` class defines an approximate surrogate model or response surface which can be used to predict the model response and its uncertainty at points where the model has not been previously evaluated. Kriging gives the best unbiased linear predictor at the interpolated points. This class generates a model :math:`\hat{y}` that express the response as a realization of regression model and Gaussian random process as:

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

The :class:`.Kriging` class offers a variety of built-in regression models, specified by the `regression` input described below.


Ordinary Kriging
~~~~~~~~~~~~~~~~~~~~

In ordinary Kriging, the regression model is assumed to take a constant value such that

.. math:: \mathcal{F}(\beta, x) = \beta_0


Universal Kriging
~~~~~~~~~~~~~~~~~~~~~

In universal Kriging, the regression model is assumed to take a general functional form. The :class:`.Kriging` class
currenly supports two univeral Kriging models, the linear regression model given by:

.. math:: \mathcal{F}(\beta, x) = \beta_0 = \sum_{i=1}^d \beta_i x_i

and the quadratic regression model given by:

.. math:: \mathcal{F}(\beta, x) = \beta_0 = \sum_{i=1}^d \beta_i x_i + \sum_{i=1}^d \sum_{j=1}^d \beta_{ij} x_i x_j


User-Defined Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new regression model to the :class:`.Kriging` class is straightforward. This is done by creating a new class
that evaluates the basis functions and the Jacobian, by extending the :class:`.Regression`.


.. autoclass:: UQpy.surrogates.kriging.Regression
    :members:

This class may be passed directly as an object to the `regression input of the :class:`.Kriging` class.
This new class must have a method ``r(self,s)`` that takes as input the samples points at which to evaluate the model and return two arrays containing the value of the basis functions and the Jacobian at these sample points.

The first output of this function should be a two dimensional numpy array with the first dimension being the number of samples and the second dimension being the number of basis functions.

The second output (i.e. Jacobian of basis function) is a three dimensional numpy array with the first dimension being the number of samples, the second dimension being the number of variables and the third dimension being the number of basis functions.

An example user-defined model is given below:


>>> class UserRegression(Regression):
>>>
>>>    def r(self, s):
>>>        s = np.atleast_2d(s)
>>>        fx = np.ones([np.size(s, 0), 1])
>>>        jf = np.zeros([np.size(s, 0), np.size(s, 1), 1])
>>>        return fx, jf


Correlation Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Kriging` class offers a variety of built-in correlation models, specified by the `correlation_model` input described below.

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


Adding a new correlation model to the :class:`.Kriging` class is straightforward. This is done by creating a new class that extends the :class:`.Correlation` abstract base class.
This requires a method takes as input the new points, training points, hyperparameters and two indicators for the computation of the derivative of correlation matrix (i.e. dt and dx).
This method evaluates the correlation matrix, its derivative with respect to the variables and its derivative with respect to the hyperparameters.

.. autoclass:: UQpy.surrogates.kriging.Correlation
    :members:

If both indicators are false, then the method should return correlation matrix, i.e. a 2-D array with first dimension being the number of points and second dimension being the number of training points.

If `dx` parameter is True, the method should return the derivative of the correlation matrix respect to the variables, i.e. a 3-D array with first dimension being the number of points, second dimension being the number of training points and third dimension being the number of variables.

If `dt` is True, then the method should return the correlation matrix and it's derivative with respect to the hyperparameters, i.e. a 3-D array with first dimension being the number of points, second dimension being the number of training points and third dimension being the number of variables.

An example user-defined model is given below:


>>> class Gaussian(Correlation):
>>>
>>>    def c(self, x, s, params, dt=False, dx=False):
>>>        stack = Correlation.check_samples_and_return_stack(x, s)
>>>        rx = np.exp(np.sum(-params * (stack ** 2), axis=2))
>>>        if dt:
>>>            drdt = -(stack ** 2) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
>>>            return rx, drdt
>>>        if dx:
>>>            drdx = - 2 * params * stack * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
>>>            return rx, drdx
>>>        return rx

Kriging Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
"""""""
.. autoclass:: UQpy.surrogates.kriging.Kriging
    :members: fit, predict, jacobian

Attributes
""""""""""
.. autoattribute:: UQpy.surrogates.kriging.Kriging.beta
.. autoattribute:: UQpy.surrogates.kriging.Kriging.err_var
.. autoattribute:: UQpy.surrogates.kriging.Kriging.C_inv