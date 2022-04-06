Grassmann Kernels
-----------------------------------

Grassmannian Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GrassmannianKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.baseclass.GrassmannianKernel import GrassmannianKernel

.. autoclass:: UQpy.utilities.kernels.baseclass.GrassmannianKernel
    :members: calculate_kernel_matrix

Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProjectionKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.ProjectionKernel import ProjectionKernel

One can use the following command to instantiate the class :class:`.ProjectionKernel`

.. autoclass:: UQpy.utilities.kernels.ProjectionKernel
    :members:

.. autoattribute:: UQpy.utilities.kernels.ProjectionKernel.kernel_matrix



Binet-Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BinetCauchyKernel` class is imported using the following command:

>>> from UQpy.utilities.kernels.BinetCauchyKernel import BinetCauchyKernel

One can use the following command to instantiate the class :class:`.BinetCauchyKernel`

.. autoclass:: UQpy.utilities.kernels.BinetCauchyKernel
    :members:

.. autoattribute:: UQpy.utilities.kernels.BinetCauchyKernel.kernel_matrix


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