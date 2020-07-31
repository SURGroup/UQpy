.. _transformations_doc:

Transformations
================

.. automodule:: UQpy.Transformations

Nataf
-----------------

``Nataf`` is   a   class   for   transforming   random   variables   using   the   `Nataf` transformation  and  calculating  the  correlation  distortion.    According  to  the Nataf transformation theory ([1]_, [2]_), a `n`-dimensional dependent random vector :math:`\textbf{X}=[X_1,...,X_n]` for which the  marginal cumulative distributions :math:`F_i(x_i)`  and the correlattion matrix :math:`\textbf{C}_x=[\xi_{ij}]` are known, can be transformed (component-wise) to standard normal random vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]` through the transformation:

.. math:: Z_{i}=\Phi^{-1}(F_i(X_{i})) 

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function. 

This transformation causes a `correlation distortion`; the correlation coefficient between the standard normal variables :math:`Z_i` and :math:`Z_j`, denoted :math:`\rho_{ij}`, is not equal to its counterpart in the parameter space (:math:`\rho_{ij} \neq \xi_{ij}`). 


If the Gaussian correlation :math:`\rho_{ij}` is know, the non-Gaussian correlation :math:`\xi_{ij}` can be determined through the following integral equation:

.. math:: \xi_{ij} =\dfrac{1}{\sigma_{X_i}\sigma_{X_j}}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\left(X_i-\mu_{X_i}\right)\left(X_j-\mu_{X_j}\right)\phi_2(Z_i,Z_j;\rho_{ij})dZ_idZ_j

where :math:`X_i =F_i^{-1}(\Phi(Z_{i}))` and :math:`\phi_2(\cdot)` is the bivariate standard normal probability density function. The integration is directly evaluated using a quadratic two-dimensional Gauss-Legendre integration. However, in the case where the non-Gaussian correlation is known :math:`\xi_{ij}`, the integral above cannot be inverted to solve for the Gaussian correlation :math:`\rho_{ij}`. In such case, iterative methods must be employed such as the Iterative Translation Approximation Method (ITAM) [3]_, used in ``UQpy``.

The Jacobian of the transformation can be also estimated with the ``Nataf`` class as:

.. math:: \mathbf{J_{xz}} = \dfrac{\partial\mathbf{x}}{\partial\mathbf{z}} =\left[\dfrac{\phi(Z_i)}{f_i(X_i)}\right]\mathbf{H}.

where :math:`\textbf{H}` is the lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
(:math:`\mathbf{C_Z}`). 

The ``Nataf`` class also allows for the inverse of the Nataf transformation, i.e. transforming a vector of standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` to random variables with prescribed marginal cumulative distributions and correlation matrix :math:`\textbf{C}_x=[\rho_{ij}]` according to:

.. math:: X_{i}=F_i^{-1}(\Phi(Z_{i}))

Nataf Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Transformations.Nataf
    :members: 


Correlate
-----------------

``Correlate`` is a class to induce correlation to an uncorrelated standard normal vector :math:`\textbf{u}=[U_1,...,U_n]`, given the correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The correlated standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` can be calculated as: 

.. math:: \mathbf{z}^\intercal = \mathbf{H}\mathbf{u}^\intercal

where :math:`\mathbf{H}` is the lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix, i.e. :math:`\mathbf{C_z}=\mathbf{H}\mathbf{H}^\intercal`. 

Correlate Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Transformations.Correlate
    :members:
	
Decorrelate
-----------------

``Decorrelate`` is a class to remove correlation from an correlated standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The uncorrelated standard normal vector :math:`\textbf{u}=[U_1,...,U_n]` can be calculated as: 

.. math:: \textbf{u}^\intercal = \mathbf{H}^{-1} \mathbf{z}^\intercal

Decorrelate Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.Transformations.Decorrelate
    :members:


**References:**

.. [1] A. Nataf, Determination des distributions dont les marges sont donnees, C. R. Acad. Sci.
   vol. 225, pp. 42-43, Paris, 1962.
.. [2] R. Lebrun and A. Dutfoy, An innovating analysis of the Nataf transformation from the copula viewpoint,
   Prob. Eng. Mech.,  vol. 24, pp. 312-320, 2009.
.. [3] Kim, H. and Shields, M.D. (2015). "Modeling strongly non-Gaussian non-stationary stochastic processes using the Iterative Translation Approximation Method and Karhunen-Loeve Expansion," Computers and Structures. 161: 31-42.



	
	