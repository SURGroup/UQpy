Polynomial Chaos Expansion - PCE
----------------------------------------

Polynomial Chaos Expansions (PCE) represent a class of methods which employ orthonormal polynomials to construct
approximate response surfaces (metamodels or surrogate models) to identify a mapping between inputs and outputs of a
numerical model :cite:`PCE1`. :class:`.PolynomialChaosExpansion` methods can be directly used for moment estimation and sensitivity analysis (Sobol indices).
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different families of univariate polynomials can be used for the PCE method. These polynomials must always be orthonormal
with respect to the arbitrary distribution. In UQpy, two families of polynomials are currently available that can be
used from their corresponding classes, namely the :class:`.Legendre` and :class:`.Hermite` polynomial class, appropriate for
data generated from a Uniform and a Normal distribution respectively.

.. toctree::
   :maxdepth: 1

    Polynomials <polynomials>


Calculation of the PCE coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Several methods exist for the calculation of the PCE coefficients. In UQpy, three non-intrusive methods can be used,
namely the Least Squares regression (:class:`.LeastSquaresRegression` class), the LASSO regression
(:class:`.LassoRegression` class) and Ridge
regression (:class:`.RidgeRegression` class) methods.

.. toctree::
   :maxdepth: 1

    Regressions <regressions>



Error Estimation
^^^^^^^^^^^^^^^^

The :meth:`.PolynomialChaosExpansion.validation_error` method can be used to estimate the accuracy of the PCE predictor.
Here, we compute the generalization error  in the form of the relative mean squared error normalized by the model variance.
The user must create an independent validation dataset :math:`[x_{val}, y_{val} = M(x_{val})]`
(i.e. a set of inputs and outputs of the computational model). The validation error is computed as

.. math:: \epsilon_{val} = \frac{N-1}{N} \Bigg[\frac{\sum_{i=1}^{N} (M(x_{val}^{(i)}) - M^{PCE}(x_{val}^{(i)}) )^{2} }{\sum_{i=1}^{N} (M(x_{val}^{(i)}) - \hat{\mu}_{Y_{val}})^{2}} \Bigg]

where :math:`\hat{\mu}_{Y_{val}}` is the sample mean value of the validation dataset output.

In case where the computational model is very expensive, the use of an alternative error measure is recommended,
for example the cross-validation error which partitions the existing training dataset into subsets and computes the
error as the average of the individual errors of each subset.



Moment Estimation
^^^^^^^^^^^^^^^^^

The :meth:`.PolynomialChaosExpansion.get_moments` method can be used for the calculation of the first two moments of the PCE model directly from the PCE coefficients. This is possible due to the orthonormality of the polynomial basis.

The first moment (mean value) is calculated as

.. math:: \mu_{PCE} = \mathbb{E} [ \mathcal{M}^{PCE}(x)] = y_{0}

where :math:`y_{0}` is the first PCE coefficient associated with the constant term.

The second moment (variance) is calculated as

.. math:: \sigma^{2}_{PCE} = \mathbb{E} [( \mathcal{M}^{PCE}(x) - \mu_{PCE} )^{2} ] = \sum_{i=1}^{p} y_{i}

where :math:`p` is the number of polynomials (first PCE coefficient is excluded).
