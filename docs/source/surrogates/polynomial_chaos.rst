Polynomial Chaos Expansion - PCE
----------------------------------------

Polynomial Chaos Expansions (PCE) represent a class of methods which employ orthonormal polynomials to construct
approximate response surfaces (metamodels or surrogate models) to identify a mapping between inputs and outputs of a
numerical model [2]_. :class:`.PolynomialChaosExpansion` methods can be directly used for moment estimation and sensitivity analysis (Sobol indices).
A PCE object can be instantiated from the class :class:`.PolynomialChaosExpansion`. The method can be used for models of both one-dimensional and multi-dimensional outputs.

Let us consider a computational model :math:`Y = \mathcal{M}(x)`, with :math:`Y \in \mathbb{R}` and a random vector with independent components :math:`X \in \mathbb{R}^M` described by the joint probability density function :math:`f_X`. The polynomial chaos expansion of :math:`\mathcal{M}(x)` is

.. math:: Y = \mathcal{M}(x) = \sum_{\alpha \in \mathbb{N}^M} y_{\alpha} \Psi_{\alpha} (X)

where the :math:`\Psi_{\alpha}(X)` are multivariate polynomials orthonormal with respect to :math:`f_X` and :math:`y_{\alpha} \in \mathbb{R}` are the corresponding coefficients.

Practically, the above sum needs to be truncated to a finite sum so that :math:`\alpha \in A` where :math:`A \subset \mathbb{N}^M`. The polynomial basis :math:`\Psi_{\alpha}(X)` is built from a set of *univariate orthonormal polynomials* :math:`\phi_j^{i}(x_i)` which satisfy the following relation

.. math:: \Big< \phi_j^{i}(x_i),\phi_k^{i}(x_i) \Big> = \int_{D_{X_i}} \phi_j^{i}(x_i),\phi_k^{i}(x_i) f_{X_i}(x_i)dx_i = \delta_{jk}

The multivariate polynomials :math:`\Psi_{\alpha}(X)` are assembled as the tensor product of their univariate counterparts as follows

.. math:: \Psi_{\alpha}(X) = \prod_{i=1}^M \phi_{\alpha_i}^{i}(x_i)

which are also orthonormal.


PCE Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion
    :members:

Univariate Orthonormal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different families of univariate polynomials can be used for the PCE method. These polynomials must always be orthonormal
with respect to the arbitrary distribution. In UQpy, two families of polynomials are currently available that can be
used from their corresponding classes, namely the :class:`.Legendre` and :class:`.Hermite` polynomial class, appropriate for
data generated from a Uniform and a Normal distribution respectively.

Polynomials Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials
    :members:

Legendre Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.Legendre
    :members:

Hermite Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.polynomials.Hermite
    :members:


Calculation of the PCE coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several methods exist for the calculation of the PCE coefficients. In UQpy, three non-intrusive methods can be used,
namely the Least Squares regression (:class:`.LeastSquaresRegression` class), the LASSO regression
(:class:`.LassoRegression` class) and Ridge
regression (:class:`.RidgeRegression` class) methods.


Least Squares Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Least Squares regression is a method for estimating the parameters of a linear regression model. The goal is to minimize the sum of squares of the differences of the observed dependent variable and the predictions of the regression model. In other words, we seek for the vector :math:`\beta`, that approximatively solves the equation :math:`X \beta \approx y`. If matrix :math:`X` is square then the solution is exact.

If we assume that the system cannot be solved exactly, since the number of equations :math:`n` is not equal to the number of unknowns :math:`p`, we are seeking the solution that is associated with the smallest difference between the right-hand-side and left-hand-side of the equation. Therefore, we are looking for the solution that satisfies the following

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \| y - X \beta \|_{2}

where :math:`\| \cdot \|_{2}` is the standard :math:`L^{2}` norm in the :math:`n`-dimensional Eucledian space :math:`\mathbb{R}^{n}`. The above function is also known as the cost function of the linear regression.

The equation may be under-, well-, or over-determined. In the context of Polynomial Chaos Expansion (PCE) the computed
vector corresponds to the polynomial coefficients. The above method can be used from the class :class:`.LeastSquaresRegression`.


LeastSquares Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression
    :members:


Lasso Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A drawback of using Least Squares regression for calculating the PCE coefficients, is that this method considers all the
features (polynomials) to be equally relevant for the prediction. This technique often results to overfitting and
complex models that do not have the ability to generalize well on unseen data. For this reason, the Least Absolute
Shrinkage and Selection Operator or LASSO can be employed (from the :class:`LassoRegression` class). This method,
introduces an :math:`L_{1}` penalty term (which encourages sparsity) in the loss function of linear regression as follows

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \{ \frac{1}{N} \| y - X \beta \|_{2} + \lambda \| \beta \|_{1} \}


where :math:`\lambda` is called the regularization strength.

Parameter :math:`\lambda` controls the level of penalization. When it is close to zero, Lasso regression is identical to Least Squares regression, while in the extreme case when it is set to be infinite all coefficients are equal to zero.

The Lasso regression model needs to be trained on the data, and for this gradient descent is used for the optimization of coefficients. In gradient descent, the gradient of the loss function with respect to the weights/coefficients :math:`\nabla Loss_{\beta}` is used and deducted from :math:`\beta^{i}` at each iteration as follows

.. math:: \beta^{i+1} = \beta^{i} - \epsilon \nabla Loss_{\beta}^{i}

where :math:`i` is the iteration step, and :math:`\epsilon` is the learning rate (gradient descent step) with a value larger than zero.


Lasso Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.LassoRegression
    :members:


Ridge Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge regression (also known as :math:`L_{2}` regularization) is another variation of the linear regression method and a special case of the Tikhonov regularization. Similarly to the Lasso regression, it introduces an additional penalty term, however Ridge regression uses an :math:`L_{2}` norm in the loss function as follows

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \{ \frac{1}{N} \| y - X \beta \|_{2} + \lambda \| \beta \|_{2} \}

where :math:`\lambda` is called the regularization strength.

Due to the penalization of terms, Ridge regression constructs models that are less prone to overfitting. The level of
penalization is similarly controlled by the hyperparameter :math:`\lambda` and the coefficients are optimized with
gradient descent. The Ridge regression method can be used from the `.RidgeRegression` class.


PolyChaosRidge Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.RidgeRegression
    :members:


Error Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.ErrorEstimation` class can be used to estimate the accuracy of the PCE predictor. Here, we compute the generalization error  in the form of the relative mean squared error normalized by the model variance. The user must create an independent validation dataset :math:`[x_{val}, y_{val} = M(x_{val})]` (i.e. a set of inputs and outputs of the computational model). The validation error is computed as

.. math:: \epsilon_{val} = \frac{N-1}{N} \Bigg[\frac{\sum_{i=1}^{N} (M(x_{val}^{(i)}) - M^{PCE}(x_{val}^{(i)}) )^{2} }{\sum_{i=1}^{N} (M(x_{val}^{(i)}) - \hat{\mu}_{Y_{val}})^{2}} \Bigg]

where :math:`\hat{\mu}_{Y_{val}}` is the sample mean value of the validation dataset output.

In case where the computational model is very expensive, the use of an alternative error measure is recommended, for example the cross-validation error which partitions the existing training dataset into subsets and computes the error as the average of the individual errors of each subset.


ErrorEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.ErrorEstimation
    :members:


Moment Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.MomentEstimation` class can be used for the calculation of the first two moments of the PCE model directly from the PCE coefficients. This is possible due to the orthonormality of the polynomial basis.

The first moment (mean value) is calculated as

.. math:: \mu_{PCE} = \mathbb{E} [ \mathcal{M}^{PCE}(x)] = y_{0}

where :math:`y_{0}` is the first PCE coefficient associated with the constant term.

The second moment (variance) is calculated as

.. math:: \sigma^{2}_{PCE} = \mathbb{E} [( \mathcal{M}^{PCE}(x) - \mu_{PCE} )^{2} ] = \sum_{i=1}^{p} y_{i}

where :math:`p` is the number of polynomials (first PCE coefficient is excluded).


MomentEstimation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.MomentEstimation
    :members:



.. [2] N. Lüthen, S. Marelli, B. Sudret, “Sparse Polynomial Chaos Expansions: Solvers, Basis Adaptivity and Meta-selection“, Available at arXiv:2009.04800v1 [stat.CO], 2020.
