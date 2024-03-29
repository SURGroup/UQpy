Nataf
-----------------

:class:`.Nataf` is   a   class   for   transforming   random   variables   using   the   `Nataf` transformation  and
calculating  the  correlation  distortion.    According  to  the Nataf transformation theory (:cite:`Nataf1`, :cite:`Nataf2`), an
`n`-dimensional dependent random vector :math:`\textbf{X}=[X_1,...,X_n]` for which the  marginal cumulative
distributions :math:`F_i(x_i)`  and the correlattion matrix :math:`\textbf{C}_x=[\xi_{ij}]` are known, can be
transformed (component-wise) to standard normal random vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation
matrix :math:`\textbf{C}_z=[\rho_{ij}]` through the transformation:

.. math:: Z_{i}=\Phi^{-1}(F_i(X_{i}))

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function.

This transformation causes a `correlation distortion`; the correlation coefficient between the standard normal
variables :math:`Z_i` and :math:`Z_j`, denoted :math:`\rho_{ij}`, is not equal to its counterpart in the parameter
space (:math:`\rho_{ij} \neq \xi_{ij}`).


If the Gaussian correlation :math:`\rho_{ij}` is know, the non-Gaussian correlation :math:`\xi_{ij}` can be
determined through the following integral equation:

.. math:: \xi_{ij} =\dfrac{1}{\sigma_{X_i}\sigma_{X_j}}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\left(X_i-\mu_{X_i}\right)\left(X_j-\mu_{X_j}\right)\phi_2(Z_i,Z_j;\rho_{ij})dZ_idZ_j

where :math:`X_i =F_i^{-1}(\Phi(Z_{i}))` and :math:`\phi_2(\cdot)` is the bivariate standard normal probability
density function. The integration is directly evaluated using a quadratic two-dimensional Gauss-Legendre integration.
However, in the case where the non-Gaussian correlation is known :math:`\xi_{ij}`, the integral above cannot be
inverted to solve for the Gaussian correlation :math:`\rho_{ij}`. In such case, iterative methods must be employed
such as the Iterative Translation Approximation Method (ITAM) :cite:`StochasticProcess13`, used in :py:mod:`UQpy`.

The Jacobian of the transformation can be also estimated with the :py:mod:`UQpy` class as:

.. math:: \mathbf{J_{xz}} = \dfrac{\partial\mathbf{x}}{\partial\mathbf{z}} =\left[\dfrac{\phi(Z_i)}{f_i(X_i)}\right]\mathbf{H}.

where :math:`\textbf{H}` is the lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
(:math:`\mathbf{C_Z}`).

The :class:`.Nataf` class also allows for the inverse of the Nataf transformation, i.e. transforming a vector of standard normal
vector :math:`\textbf{z}=[Z_1,...,Z_n]` to random variables with prescribed marginal cumulative distributions and correlation
matrix :math:`\textbf{C}_x=[\rho_{ij}]` according to:

.. math:: X_{i}=F_i^{-1}(\Phi(Z_{i}))

Nataf Class
^^^^^^^^^^^^^^

The :class:`.Nataf` class is imported using the following command:

>>> from UQpy.transformations.Nataf import Nataf

Methods
"""""""
.. autoclass:: UQpy.transformations.Nataf
    :members: run, itam, distortion_z2x, rvs

Attributes
""""""""""
.. autoattribute:: UQpy.transformations.Nataf.samples_x
.. autoattribute:: UQpy.transformations.Nataf.samples_z
.. autoattribute:: UQpy.transformations.Nataf.jzx
.. autoattribute:: UQpy.transformations.Nataf.jxz
.. autoattribute:: UQpy.transformations.Nataf.corr_z
.. autoattribute:: UQpy.transformations.Nataf.corr_x
.. autoattribute:: UQpy.transformations.Nataf.H


Examples
""""""""""

.. toctree::

   Nataf Examples <../auto_examples/transformations/nataf/index>