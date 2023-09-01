Gaussian Process Regression
---------------------------------------

The :class:`.GaussianProcessRegression` class defines an approximate surrogate model or response surface which can be used to predict the model response and its uncertainty at points where the model has not been previously evaluated. Gaussian Process regression utilizes the concepts of Bayesian modelling and identifies a suitable function, which fits the training data (:math:`X \in \mathbb{R}^{n \times d}, Y \in \mathbb{R}^{n, 1}`). First, it formulates a prior function over the output variable (:math:`y=g(x))`) as a Gaussian Process, which can be defined using a mean function and a kernel.

.. math:: p(y|x, \theta) = \mathcal{N}(m(x), K(x, x))

The mean function :math:`m(x)` is given as a linear combination of ':math:`p`' chosen scalar basis functions as:

.. math:: m(\beta, x) = \mathbb{E}[g(x)] = \beta_1 f_1(x) + \dots + \beta_p f_p(x) = f(x)^T \beta=f(x)^T \beta.

The kernel :math:`k(x, s)` defines the covariance function as:

.. math:: k(x, s) = \mathbb{E}[(g(x)-m(x))(g(s)-m(s))]

and the Gaussian process can be written as,

.. math:: y = g(x) \sim \mathcal{GP}(m(x, \beta), k(x,s, \theta)),

where, :math:`\beta` is the regression coefficient estimated by least square solution and :math:`\theta=\{l_1, ..., l_d, \sigma \}` are a set of hyperparameters generally governing the correlation length (lengthscale, :math:`l_i`) and the process variance (:math:`\sigma`) of the model, determined by maximixing the log-likelihood function

.. math:: \text{log}(p(y|x, \theta)) = -\frac{1}{2}(Y-F\beta)^T K^{-1} (Y-F\beta) - \frac{1}{2}\text{log}(|K|) - \frac{n}{2}\text{log}(2\pi)


The covariance is evaluated between a set of existing sample points :math:`X` in the domain of interest to form the covariance matrix :math:`K=K(X, X)`, and the basis functions are evaluated at the sample points :math:`X` to form the matrix :math:`F`. Using these matrices, the regression coefficients, :math:`\beta`, is computed as

.. math:: (F^T K^{-1} F)\beta^* = F^T K^{-1} Y

The joint distribution between the training outputs (:math:`g(X)`) and test outputs (:math:`g(X^*)`) can be expressed as:

.. math:: \begin{bmatrix} Y \\ g(X^*)\end{bmatrix} = \mathcal{N} \Bigg(\begin{bmatrix} m(X) \\ m(X^*)\end{bmatrix}, \begin{bmatrix} K(X, X) & K(X, X^*)\\ K(X^*, X) & K(X^*, X^*) \end{bmatrix}  \Bigg)

The final predictor function is then given by the mean of the posterior distribution :math:`g(X^*)|Y, X, X^*`, defined as:

.. math:: \hat{g}(X^*) = f(X^*)^T \beta^* + K(X, X^*)^T K^{-1}(Y - F\beta^*)

and the covariance matrix of the posterior distribution is expressed as:

.. math:: cov(g(X^*)) = K(X^*, X^*) - K(X^*, X)K^{-1}K(X, X^*)

In case of noisy output (i.e. :math:`y = g(x)+\epsilon`), where noise :math:`\epsilon` is a independent gaussian distribution with variance :math:`\sigma_n^2`. The :class:`.GaussianProcessRegression` class includes noise standard deviation in the hyperparameters (:math:`\theta=\{l_1, ..., l_d, \sigma, \sigma_n \}`) along with the lengthscales and process standard deviation, and identify them by maximixing log-likelihood function. The mean and covariance of the posterior distribution is modified by substituting :math:`K` as :math:`K+\sigma_n^2 I`:

.. math:: \hat{g}(X^*) = f(X^*)^T \beta^* + K(X, X^*)^T (K+\sigma_n^2 I)^{-1}(Y - F\beta^*) \\ cov(g(X^*)) = K(X^*, X^*) - K(X^*, X)(K+\sigma_n^2 I)^{-1}K(X, X^*)

Regression Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GaussianProcessRegression` class offers a variety of built-in regression models, specified by the `regression` input described below.


Constant Model
"""""""""""""""""""

The :class:`.ConstantRegression` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.regression_models.ConstantRegression import ConstantRegression


In Constant model, the regression model is assumed to take a constant value such that

.. math:: f(\beta, x) = \beta_0


Linear Model
""""""""""""""""""""

The :class:`.LinearRegression` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression


The regression model is defined by the linear basis function on each input dimension.

.. math:: f(\beta, x) = \beta_0 + \sum_{i=1}^d \beta_i x_i

Quadratic Model
""""""""""""""""""""

The :class:`.QuadraticRegression` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.regression_models.QuadraticRegression import QuadraticRegression

The quadratic regression model is given by:

.. math:: f(\beta, x) = \beta_0 + \sum_{i=1}^d \beta_i x_i + \sum_{i=1}^d \sum_{j=1}^d \beta_{ij} x_i x_j


User-Defined Regression Model
"""""""""""""""""""""""""""""""

Adding a new regression model to the :class:`.GaussianProcessRegressor` class is straightforward. This is done by creating a new class
that evaluates the basis functions, by extending the :class:`.Regression`.

The :class:`.Regression` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.regression_models.baseclass.Regression import Regression

.. autoclass:: UQpy.surrogates.gaussian_process.Regression
    :members:

This class may be passed directly as an object to the regression input of the :class:`.GaussianProcessRegression` class.
This new class must have a method ``r(self,s)`` that takes as input the samples points at which to evaluate the model
and return the value of the basis functions at these sample points.

The output of this function should be a two dimensional numpy array with the first dimension being the number of
samples and the second dimension being the number of basis functions.

An example user-defined model is given below:


>>> class UserRegression(Regression):
>>>
>>>    def r(self, s):
>>>        s = np.atleast_2d(s)
>>>        fx = np.ones([np.size(s, 0), 1])
>>>        return fx

Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GaussianProcessRegression` class offers a variety of built-in kernels, specified by the `kernel` input described below.

Radial Basis Function Kernel
"""""""""""""""""""""""""""""

The :class:`.RBF` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

The RBF kernel takes the following form:

.. math:: K(h_i, \theta_i) = \sigma^2 \prod_{1}^{d} \mathcal{R}_i(h_i, l_i) = \sigma^2 \prod_{1}^{d} \exp\bigg[ -\frac{h_i^2}{2l_i^2}\bigg]

where :math:`h_i = s_i-x_i`.

Matern Kernel
"""""""""""""""""""""""""""""

The :class:`.Matern` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.kernels.Matern import Matern

The Matern kernel takes the following form:

.. math:: K(x, s, \theta) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \bigg( \sqrt{2 \nu} \frac{d}{l} \bigg)^{\nu} K_v \bigg(\sqrt{2 \nu} \frac{d}{l} \bigg)

where :math:`d = ||x-s||_2^{1/2}` is the euclidean distance and :math:`\theta` is consist of lengthscale (:math:`l`), process variance (:math:`\sigma^2`) and smoothing parameter (:math:`\nu`). Also, :math:`\Gamma` is tha gamma function and :math:`K_v` is the modified Bessel function. This kernel concides with absolute exponential and RBF kernel for :math:`\nu=0.5` and :math:`\infty`, respectively.

User-Defined Kernel
""""""""""""""""""""""""""""

Adding a new kernel to the :class:`.GaussianProcessRegression` class is straightforward. This is done by creating a new class
that extends the :class:`.Kernel` abstract base class.
This new class must have a method ``c(self, x, s, params)`` that takes as input the new points, training points and hyperparameters.
Notice that the input ``params`` include lengthscales and process standard deviation, not noise standard deviation (even for noisy data).

The :class:`.Kernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.Kernel import Kernel

The method should return covariance matrix, i.e. a 2-D array with first dimension
being the number of new points and second dimension being the number of training points.

An example user-defined kernel is given below:


>>> class RBF(Kernel):
>>>
>>>    def c(self, x, s, params):
>>>         l, sigma = params[:-1], params[-1]
>>>         stack = cdist(x/l, s/l, metric='euclidean')
>>>         cx = sigma**2 * np.exp(-(stack**2)/2)
>>>         return cx

Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GaussianProcessRegression` class offers a built-in constraints on the hyperparameters estimation while maximixing loglikelihood function.

NonNegative
"""""""""""""""""""""""""""""

The :class:`.NonNegative` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.constraints.NonNegative import NonNegative

The NonNegative constraints enforces non-negativity in the GP model, this is useful for treating many physical properties as output variable. This constraint improves prediction and error estimates, where model output is always greater than or equal to zero. This is ensured by solving the maximum likelihood problem with the following constraints.

.. math:: \mathrm{argmax}_{\theta} \log{p(y|x, \theta)} \quad \quad \quad  \text{s.t.   } \begin{matrix} \hat{g}(X_c) - z\sigma_{\hat{g}}(X_c) \geq 0 \\ |Y-\hat{g}(X)| \geq \epsilon \end{matrix}

where :math:`X_c` is the set of constraint points defined by user, :math:`z` is the number of standard deviation below mean, that is non-negative for constraint points. :math:`\epsilon` is the error tolerance between output value and prediction at the sample points.

User-Defined Constraints
""""""""""""""""""""""""""""

Adding a new kernel to the :class:`.GaussianProcessRegression` class is straightforward. This is done by creating a new class
that extends the :class:`UQpy.surrogates.gaussian_process.kernels.baseclass.Constraints` abstract base class.
This new class must have two methods ``define_arguments(self, x_train, y_train, predict_function)`` and ``constraints(theta, kwargs)``.

The :class:`UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints import ConstraintsGPR

.. autoclass:: UQpy.surrogates.gaussian_process.ConstraintsGPR
    :members:

The first method takes the sample points, output values and prediction method as inputs. This method combines the input attributes
(defined while initiating the new class) and inputs of the ``define_arguments`` method. This method returns a list containing a dictionary of all the arguments to the constraints function.

The second method ``constraints`` is static method, which takes first input as the log-transformed hyperparameters (:math:`\theta`) and second input is a dictionary containing all arguments as defined in the first method (``define_arguments``).
User can unpacked the stored variables in the dictionary and return the constraints function. Notice that this is a static method as optimizers require constraints to be a callable.

The constraints on the hyperparameters are only compatible with ``FminCobyla`` class as optimizer.

An example user-defined kernel is given below:


>>> class NonNegative(ConstraintsGPR):
>>>     def __init__(self, constraint_points, observed_error=0.01, z_value=2):
>>>         self.constraint_points = constraint_points
>>>         self.observed_error = observed_error
>>>         self.z_value = z_value
>>>         self.kwargs = {}
>>>         self.constraint_args = None
>>>
>>>     def define_arguments(self, x_train, y_train, predict_function):
>>>         self.kwargs['x_t'] = x_train
>>>         self.kwargs['y_t'] = y_train
>>>         self.kwargs['pred'] = predict_function
>>>         self.kwargs['const_points'] = self.constraint_points
>>>         self.kwargs['obs_err'] = self.observed_error
>>>         self.kwargs['z_'] = self.z_value
>>>         self.constraint_args = [self.kwargs]
>>>         return self.constraints
>>>
>>>     @staticmethod
>>>     def constraints(theta_, kwargs):
>>>         x_t, y_t, pred = kwargs['x_t'], kwargs['y_t'], kwargs['pred']
>>>         const_points, obs_err, z_ = kwargs['const_points'], kwargs['obs_err'], kwargs['z_']
>>>
>>>         tmp_predict, tmp_error = pred(const_points, True, hyperparameters=10**theta_)
>>>         constraint1 = tmp_predict - z_ * tmp_error
>>>
>>>         tmp_predict2 = pred(x_t, False, hyperparameters=10**theta_)
>>>         constraint2 = obs_err - np.abs(y_t[:, 0] - tmp_predict2)
>>>
>>>         constraints = np.concatenate((constraint1, constraint2), axis=None)
>>>         return constraints

GaussianProcessRegression Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GaussianProcessRegression` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression

Methods
"""""""
.. autoclass:: UQpy.surrogates.gaussian_process.GaussianProcessRegression
    :members: fit, predict

Attributes
""""""""""
.. autoattribute:: UQpy.surrogates.gaussian_process.GaussianProcessRegression.beta
.. autoattribute:: UQpy.surrogates.gaussian_process.GaussianProcessRegression.err_var
.. autoattribute:: UQpy.surrogates.gaussian_process.GaussianProcessRegression.C_inv

Examples
""""""""""

.. toctree::

   Gaussian Process Regression Examples <../auto_examples/surrogates/gpr/index>