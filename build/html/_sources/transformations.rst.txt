.. _transformations:

Transformations
================

This module contains functionality for all the transformations methods supported in ``UQpy``. 
The module currently contains the following classes:

- ``Nataf``: Class to perform the Nataf isoprobabilistic transformations.
- ``Correlate``: Class to induce correlation to a standard normal vector.
- ``Decorrelate``: Class to remove correlation from a standard normal vector.


Nataf
-----------------

``Nataf`` is   a   class   for   transforming   random   variables   using   the   `Nataf` transformation  and  calculating  the  correlation  distortion.    According  to  the Nataf transformation theory ([1]_, [2]_), a `n`-dimensional dependent random vector :math:`\textbf{x}=[X_1,...,X_n]` for which the  marginal cumulative distributions :math:`F_i(x_i)`  and the correlattion matrix :math:`\textbf{C}_x=[\xi_{ij}]` are known, can be transformed (component-wise) to standard normal random vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]` through the transformation:

.. math:: Z_{i}=\Phi^{-1}(F_i(X_{i})) 

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function. 

This transformation causes a `correlation distortion`; the correlation coefficient between the standard normal variables :math:`Z_i` and :math:`Z_j` is not equal to its counterpart in the  random space (:math:`\rho_{ij} \neq \xi_{ij}`). In this case :math:`\rho_{ij}` can be determined through the following integral equation:

.. math:: \xi_{ij} =\dfrac{1}{\sigma_{X_i}\sigma_{X_j}}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\left(X_i-\mu_{X_i}\right)\left(X_j-\mu_{X_j}\right)\phi_2(Z_i,Z_j;\rho_{ij})dZ_idZ_j

where :math:`X_i =F_i^{-1}(\Phi(Z_{i}))` and :math:`\phi_2(\cdot)` is the bivariate standard normal probability density function. The integration is directly evaluated using a quadratic two-dimensional Gauss-Legendre integration.

The Jacobian of the transformation can be also estimated with the ``Nataf`` class as:

.. math:: \mathbf{J_{xz}} = \dfrac{\partial\mathbf{x}}{\partial\mathbf{z}} =\left[\dfrac{\phi(Z_i)}{f_i(X_i)}\right]\mathbf{H}.

where :math:`\textbf{H}` is the lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
(:math:`\mathbf{C_Z}`). The 'Nataf' class also allows for the inverse of the Nataf transformation, i.e. transforming a vector of standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` to random variables with prescribed marginal cumulative distributions and correlation matrix :math:`\textbf{C}_x=[\rho_{ij}]` according to:

.. math:: X_{i}=F_i^{-1}(\Phi(Z_{i}))

The inverse Nataf is widely-used in reliabilityt analysis using FORM.

The ``Nataf``   class can be imported in a python script using the following command:

>>> from UQpy.Transformations import Nataf

.. autoclass:: UQpy.Transformations.Nataf
    :members: 


Correlate
-----------------

``Correlate`` is a class to induce correlation to an uncorrelated standard normal vector :math:`\textbf{u}=[U_1,...,U_n]`, given the correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The correlated standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` can be calculated as: 

.. math:: \mathbf{z}^\intercal = \mathbf{H}\mathbf{u}^\intercal

where :math:`\mathbf{H}` is the lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix, i.e. :math:`\mathbf{C_z}=\mathbf{H}\mathbf{H}^\intercal`. 

.. autoclass:: UQpy.Transformations.Correlate
    :members:
	
Decorrelate
-----------------

``Decorrelate`` is a class to remove correlation from an correlated standard normal vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]`. The uncorrelated standard normal vector :math:`\textbf{u}=[U_1,...,U_n]` can be calculated as: 

.. math:: \textbf{u}^\intercal = \mathbf{H}^{-1} \mathbf{z}^\intercal

The ``Decorrelate`` class can be imported in a python script using the following command:

>>> from UQpy.Transformations import Decorrelate


**References:**

.. [1] A. Nataf, Determination des distributions dont les marges sont donnees, C. R. Acad. Sci.
   vol. 225, pp. 42-43, Paris, 1962.
.. [2] R. Lebrun and A. Dutfoy, An innovating analysis of the Nataf transformation from the copula viewpoint,
   Prob. Eng. Mech.,  vol. 24, pp. 312-320, 2009.



	
	