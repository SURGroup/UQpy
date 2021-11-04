Grassmann Kernels
-----------------------------------

In several applications the use of subspaces is essential to describe the underlying geometry of data. However, it is
well-known that subspaces do not follow the Euclidean geometry because they lie on the Grassmann manifold.Therefore, working with subspaces requires the definition of an embedding structure of the Grassmann manifold into a Hilbert space. Thus, using positive definite kernels is studied as a solution to this problem. In this regard, a real-valued positive definite kernel is defined as a symmetric function :math:`k:\mathcal{X}\times \mathcal{X} \rightarrow \mathbb{R}` if and only if :math:`\sum^n_{I,j=1}c_i c_j k(x_i,x_j) \leq 0` for :math:`n \in \mathbb{N}`, :math:`x_i in \mathcal{X}` and :math:`c_i \in \mathbb{R}`. On the Grassmann manifold a kernel is defined as a well-defined and positive definite function :math:`k:\mathcal{G}(p,n)\times \mathcal{G}(p,n) \rightarrow \mathbb{R}` [1]_, [2]_. A Grassmann kernel is a well-defined positive definite function that embeds the Grassmannian into a Hilbert space. :py:mod:`UQpy` introduces two Grassmann kernels have been proposed in literature and have demonstrated the potential for subspace-based learning problems.


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

.. [2] M. T. Harandi, M. Salzmann, S. Jayasumana, R. Hartley, H. Li, Expanding the family of Grassmannian kernels: An embedding perspective, 2014, 1622014.arXiv:1407.1123.
