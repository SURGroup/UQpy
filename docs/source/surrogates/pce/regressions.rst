Calculation of the PCE coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Several methods exist for the calculation of the PCE coefficients. In UQpy, three non-intrusive methods can be used,
namely the Least Squares regression (:class:`.LeastSquaresRegression` class), the LASSO regression
(:class:`.LassoRegression` class) and Ridge
regression (:class:`.RidgeRegression` class) methods.


Least Squares Regression
"""""""""""""""""""""""""""""""""""

Least Squares regression is a method for estimating the parameters of a linear regression model. The goal is to minimize the sum of squares of the differences of the observed dependent variable and the predictions of the regression model. In other words, we seek for the vector :math:`\beta`, that approximatively solves the equation :math:`X \beta \approx y`. If matrix :math:`X` is square then the solution is exact.

If we assume that the system cannot be solved exactly, since the number of equations :math:`n` is not equal to the number of unknowns :math:`p`, we are seeking the solution that is associated with the smallest difference between the right-hand-side and left-hand-side of the equation. Therefore, we are looking for the solution that satisfies the following

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \| y - X \beta \|_{2}

where :math:`\| \cdot \|_{2}` is the standard :math:`L^{2}` norm in the :math:`n`-dimensional Eucledian space :math:`\mathbb{R}^{n}`. The above function is also known as the cost function of the linear regression.

The equation may be under-, well-, or over-determined. In the context of Polynomial Chaos Expansion (PCE) the computed
vector corresponds to the polynomial coefficients. The above method can be used from the class :class:`.LeastSquaresRegression`.


LeastSquares Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.LeastSquaresRegression` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression import LeastSquareRegression

.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression
    :members:


Lasso Regression
""""""""""""""""""""""""""""

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


Lasso Regression Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.LassoRegression` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.regressions.LassoRegression import LassoRegression

.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.LassoRegression
    :members:


Ridge Regression
"""""""""""""""""""""

Ridge regression (also known as :math:`L_{2}` regularization) is another variation of the linear regression method and a special case of the Tikhonov regularization. Similarly to the Lasso regression, it introduces an additional penalty term, however Ridge regression uses an :math:`L_{2}` norm in the loss function as follows

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \{ \frac{1}{N} \| y - X \beta \|_{2} + \lambda \| \beta \|_{2} \}

where :math:`\lambda` is called the regularization strength.

Due to the penalization of terms, Ridge regression constructs models that are less prone to overfitting. The level of
penalization is similarly controlled by the hyperparameter :math:`\lambda` and the coefficients are optimized with
gradient descent. The Ridge regression method can be used from the `.RidgeRegression` class.


Ridge Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.PceSensitivity` class is imported using the following command:

>>> from UQpy.sensitivity.PceSensitivity import PceSensitivity

.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.RidgeRegression
    :members:
    
LAR Regression
"""""""""""""""""""""


LeastAngleRegression Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.LeastAngleRegression` class is imported using the following command:

>>> from UQpy.surrogates.polynomial_chaos.regressions.LeastAngleRegression import LeastAngleRegression

.. autoclass:: UQpy.surrogates.polynomial_chaos.regressions.LeastAngleRegression
    :members:
