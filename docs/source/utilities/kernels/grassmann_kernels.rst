Grassmannian Kernels
-----------------------------------

In several applications the use of subspaces is essential to describe the underlying geometry of data. However, it is
well-known that sets of subspaces do not follow Euclidean geometry. Instead they have a Reimannian structure and lie on
a Grassmann manifold.  Grassmannian kernels can be used to embed the structure of the Grassmann manifold into a Hilbert
space. On the Grassmann manifold, a kernel is defined as a positive definite function
:math:`k:\mathcal{G}(p,n)\times \mathcal{G}(p,n) \rightarrow \mathbb{R}` :cite:`kernels_1`, :cite:`kernels_2`.

:py:mod:`UQpy` includes Grassmannian kernels through the :class:`.GrassmannianKernel` parent class,
with specific kernels included as subclasses. This is described in the following.

Grassmannian Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GrassmannianKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.GrassmannianKernel import GrassmannianKernel

.. autoclass:: UQpy.utilities.kernels.baseclass.GrassmannianKernel
    :members: calculate_kernel_matrix

Projection Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The projection kernel is defined as:

.. math:: k_p(\mathbf{X}_i, \mathbf{X}_j) = ||\mathbf{X}_i^T\mathbf{X}_j||_F^2

where :math:`\mathbf{X}_i, \mathbf{X}_j \in \mathcal{G}(p,n)`

The :class:`.ProjectionKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.ProjectionKernel import ProjectionKernel

One can use the following command to instantiate the :class:`.ProjectionKernel` class.

.. autoclass:: UQpy.utilities.kernels.ProjectionKernel
    :members:

.. autoattribute:: UQpy.utilities.kernels.ProjectionKernel.kernel_matrix



Binet-Cauchy Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Binet-Cauchy Kernel is defined as:

.. math:: k_p(\mathbf{X}_i, \mathbf{X}_j) = \det(\mathbf{X}_i^T\mathbf{X}_j)^2

where :math:`\mathbf{X}_i, \mathbf{X}_j \in \mathcal{G}(p,n)`

The :class:`.BinetCauchyKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.BinetCauchyKernel import BinetCauchyKernel

One can use the following command to instantiate the :class:`.BinetCauchyKernel` class.

.. autoclass:: UQpy.utilities.kernels.BinetCauchyKernel
    :members:

.. autoattribute:: UQpy.utilities.kernels.BinetCauchyKernel.kernel_matrix


