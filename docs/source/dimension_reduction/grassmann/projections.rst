SVD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SvdProjection` class is used to project each data point of a given dataset onto a Grassmann manifold using the Singular Value Decomposition (SVD). The SVD factorizes a matrix :math:`\mathbf{X}\in \mathbb{R}^{n \times m}` into three matrices:

.. math:: \mathbf{X} =  \mathbf{U}  \mathbf{\Sigma}  \mathbf{V}^\intercal

where :math:`\mathbf{U}` and :math:`\mathbf{V}` are the matrices of left and right eigenvectors, respectively and :math:`\mathbf{\Sigma}` is a diagonal matrix containing the eigenvalues. Since :math:`\mathbf{U}` and :math:`\mathbf{V}` are orthonormal matrices we consider them to be representatives of the data point on the Grassmann manifold. The :class:`.SvdProjection` class allows the user to define the Grassmann manifold :math:`\mathcal{G}(p, n)` on which the data will reside by selecting the number of :math:`p-` planes, i.e., the rank of matrix :math:`\mathbf{U}` is equal to the number of :math:`p-` planes. It also provides the flexibility to define various compositions of Grassmann kernels using the :math:`\mathbf{U}` and :math:`\mathbf{V}` matrices.

The :class:`.SvdProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.SvdProjection import SvdProjection

A description of the class signature is shown below:

.. autoclass:: UQpy.dimension_reduction.SvdProjection
    :members:
	
------------------------------------------	

Depending on whether we want to use matrix :math:`\mathbf{U}` or :math:`\mathbf{V}` to define a kernel on the Grassmann manifold we need to use the :class:`.KernelComposition` class  which is imported as:

>>> from UQpy.dimension_reduction.KernelComposition import KernelComposition

The signature of the initializer is shown below:

.. autoclass:: UQpy.dimension_reduction.KernelComposition
    :members:

------------------------------------------------------------------------------------

The abstract :class:`.ManifoldProjection` class is the parent class that allows the user to define a set of methods that
must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.ManifoldProjection
    :members: