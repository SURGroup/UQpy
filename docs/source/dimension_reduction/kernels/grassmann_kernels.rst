Grassmann Kernels
-----------------------------------

A Grassmann kernel is a well-defined positive definite function that embeds the Grassmannian into a Hilbert space. :py:mod:`UQpy` introduces two Grassmann kernels have been proposed in literature and have demonstrated the potential for subspace-based learning problems. [1]_


Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProjectionKernel` class is imported using the following command:

>>> from UQpy.dimension_reduction.kernels.grassmann.ProjectionKernel import ProjectionKernel

One can use the following command to instantiate the class :class:`.ProjectionKernel`

.. autoclass:: UQpy.dimension_reduction.kernels.grassmann.ProjectionKernel
    :members:
	
	

Binet-Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BinetCauchyKernel` class is imported using the following command:

>>> from UQpy.dimension_reduction.kernels.grassmann.BinetCauchyKernel import BinetCauchyKernel

One can use the following command to instantiate the class :class:`.BinetCauchyKernel`

.. autoclass:: UQpy.dimension_reduction.kernels.grassmann.BinetCauchyKernel
    :members:
	
	
	
.. [1] J. Hamm and D. D. Lee, "Grassmann Discriminant Analysis: a Unifying View on Subspace-Based Learning", July 2008.
