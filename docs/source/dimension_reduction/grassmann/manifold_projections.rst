Projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A collection of methods to project data on the Grassmann manifold.

The abstract :class:`.GrassmannProjection` class is the parent class that allows the user to define a set of methods that
must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.GrassmannProjection
    :members:

The :class:`.GrassmannProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.GrassmannProjection import GrassmannProjection

SVD Projection
~~~~~~~~~~~~~~~~~~~~~~

The :class:`.SvdProjection` class is used to project each data point of a given dataset onto a Grassmann manifold using the Singular Value Decomposition (SVD). The SVD factorizes a matrix :math:`\mathbf{X}\in \mathbb{R}^{n \times m}` into three matrices:

.. math:: \mathbf{X} =  \mathbf{U}  \mathbf{\Sigma}  \mathbf{V}^\intercal

where :math:`\mathbf{U}` and :math:`\mathbf{V}` are the matrices of left and right eigenvectors, respectively and :math:`\mathbf{\Sigma}` is a diagonal matrix containing the eigenvalues. Since :math:`\mathbf{U}` and :math:`\mathbf{V}` are orthonormal matrices we consider them to be representatives of the data point on the Grassmann manifold. The :class:`.SvdProjection` class allows the user to define the Grassmann manifold :math:`\mathcal{G}(p, n)` on which the data will reside by selecting the number of :math:`p-` planes, i.e., the rank of matrix :math:`\mathbf{U}` is equal to the number of :math:`p-` planes. It also provides the flexibility to define various compositions of Grassmann kernels using the :math:`\mathbf{U}` and :math:`\mathbf{V}` matrices.

The :class:`.SvdProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.SvdProjection import SvdProjection

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.SvdProjection
    :members:


Calculate Svd Projection SUM or PRODUCT kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> D1 = 6
>>> r0 = 2  # rank sample 0
>>> r1 = 3  # rank sample 1
>>> r2 = 4  # rank sample 2
>>> r3 = 3  # rank sample 2
>>>
>>> Sol0 = np.dot(np.random.rand(D1, r0), np.random.rand(r0, D1))
>>> Sol1 = np.dot(np.random.rand(D1, r1), np.random.rand(r1, D1))
>>> Sol2 = np.dot(np.random.rand(D1, r2), np.random.rand(r2, D1))
>>> Sol3 = np.dot(np.random.rand(D1, r3), np.random.rand(r3, D1))
>>>
>>> # Creating a list of solutions.
>>> Solutions = [Sol0, Sol1, Sol2, Sol3]
>>> from UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations import Grassmann
>>> manifold_projection = SvdProjection(Solutions, p="max")
>>> kernel = ProjectionKernel()
>>>
>>> kernel_psi = kernel.calculate_kernel_matrix(manifold_projection.u)
>>> kernel_phi = kernel.calculate_kernel_matrix(manifold_projection.v)
>>>
>>> sum_kernel = kernel_psi + kernel_phi
>>> product_kernel = kernel_psi * kernel_phi

