Gaussian Process Regression
---------------------------------------

The :class:`.GaussianProcessRegression` class defines an approximate surrogate model or response surface which can be used to predict the model response and its uncertainty at points where the model has not been previously evaluated. Gaussian Process regressor gives the best unbiased linear predictor at the interpolated points. This class generates a model :math:`\hat{y}` that express the response as a realization of regression model and Gaussian random process as:

.. math:: \hat{y}(x) = \mathcal{F}(\beta, x) + z(x).

The regression model (:math:`\mathcal{F}`) is given as a linear combination of ':math:`p`' chosen scalar basis functions as:

.. math:: \mathcal{F}(\beta, x) = \beta_1 f_1(x) + \dots + \beta_p f_p(x) = f(x)^T \beta.

The random process :math:`z(x)` has zero mean and its covariance is defined through the separable correlation function:

.. math:: E\big[z(s)z(x)] = \mathcal{K}(l, s, x)

where,

.. math:: \mathcal{K}(s, x; \theta) = \sigma^2 \prod_{i=1}^d \mathcal{R}_i(s_i, x_i; l_i),

and :math:`\theta=\{l_1, ..., l_d, \sigma \}` are a set of hyperparameters generally governing the correlation length (lengthscale, :math:`l_i`) and the process variance (:math:`\sigma`) of the model, determined by maximixing the log-likelihood function

.. math:: \text{log}(p(y|x, \theta)) = -\frac{1}{2}y^T \mathcal{K}^{-1} y - \frac{1}{2}\text{log}(|\mathcal{K}|) - \frac{n}{2}\text{log}(2\pi)


The correlation is evaluated between a set of existing sample points :math:`s` and points :math:`x` in the domain of interest to form the correlation matrix :math:`R`, and the basis functions are evaluated at the sample points :math:`s` to form the matrix :math:`F`. Using these matrices, the regression coefficients, :math:`\beta`, is computed as

.. math:: (F^T K^{-1} F)\beta^* = F^T K^{-1} Y

The final predictor function is then given by:

.. math:: \hat{y}(x) = f(x)^T \beta^* + k(x)^T K^{-1}(Y - F\beta^*)

Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GaussianProcessRegression` class offers a variety of built-in kernels, specified by the `kernel` input described below.

Radial Basis Function Kernel
"""""""""""""""""""""""""""""

The :class:`.RBF` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

The RBF kernel takes the following form:

.. math:: \mathcal{K}(h_i, \theta_i) = \sigma^2 \prod_{1}^{d} \mathcal{R}_i(h_i, l_i) = \sigma^2 \prod_{1}^{d} \exp\bigg[ -\frac{h_i^2}{2l_i^2}\bigg]

where :math:`h_i = s_i-x_i`.

Matern Kernel
"""""""""""""""""""""""""""""

The :class:`.Matern` class is imported using the following command:

>>> from UQpy.surrogates.gaussian_process.kernels.Matern import Matern

The Matern kernel takes the following form:

.. math:: \mathcal{K}(x, s, \theta) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \bigg( \sqrt{2 \nu} \frac{d}{l} \bigg)^{\nu} K_v \bigg(\sqrt{2 \nu} \frac{d}{l} \bigg)

where :math:`d = ||x-s||_2^{1/2}` is the euclidean distance and :math:`\theta` is consist of lengthscale (:math:`l`), process variance (:math:`\sigma^2`) and smoothing parameter (:math:`\nu`). Also, :math:`\Gamma` is tha gamma function and :math:`K_v` is the modified Bessel function. This kernel concides with exponential and RBF kernel for :math:`\nu=0.5` and :math:`\infty`, respectively.

GaussianProcessRegressor Class
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