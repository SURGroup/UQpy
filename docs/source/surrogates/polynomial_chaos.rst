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


.. toctree::
   :maxdepth: 1

    Polynomial Bases <pce/polynomial_bases>
    Polynomials <pce/polynomials>
    Regressions <pce/regressions>
    Polynomial Chaos Expansion <pce/pce>
    Physics-informed Polynomial Chaos Expansion <pce/physics_informed>












