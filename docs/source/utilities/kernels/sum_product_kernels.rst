Sum and Product Kernels
-----------------------------------

Sum and product kernels defined as

.. math:: k_{\text{sum}}(x_i, x_j) = k_1(x_i, x_j) + k_2(x_i, x_j)

and

.. math:: k_{\text{prod}}(x_i, x_j) = k_1(x_i, x_j) \cdot k_2(x_i, x_j)

respectively can be easily computed by simply summing or multiply the kernel matrices as illustrated in the following
example.

Example sum and product Grassmannian kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import numpy as np
>>> from UQpy.dimension_reduction.grassmann_manifold.projections.SVDProjection import SVDProjection
>>> from UQpy.utilities.kernels.ProjectionKernel import ProjectionKernel
>>>
>>> D1 = 6
>>> r0 = 2  # rank sample 0
>>> r1 = 3  # rank sample 1
>>> r2 = 4  # rank sample 2
>>> r3 = 3  # rank sample 2
>>> Sol0 = np.dot(np.random.rand(D1, r0), np.random.rand(r0, D1))
>>> Sol1 = np.dot(np.random.rand(D1, r1), np.random.rand(r1, D1))
>>> Sol2 = np.dot(np.random.rand(D1, r2), np.random.rand(r2, D1))
>>> Sol3 = np.dot(np.random.rand(D1, r3), np.random.rand(r3, D1))
>>>
>>> # Creating a list of solutions.
>>> Solutions = [Sol0, Sol1, Sol2, Sol3]
>>> manifold_projection = SVDProjection(Solutions, p="max")
>>> kernel = ProjectionKernel()
>>>
>>> kernel.calculate_kernel_matrix(manifold_projection.u)
>>> kernel_psi = kernel.kernel_matrix
>>>
>>> kernel.calculate_kernel_matrix(manifold_projection.v)
>>> kernel_phi = kernel.kernel_matrix
>>>
>>> sum_kernel = kernel_psi + kernel_phi
>>> product_kernel = kernel_psi * kernel_phi