Baseclass 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The abstract :class:`.ManifoldProjection` class is the parent class for all classes in :mod:`.manifold_projections` module. It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.grassmann_manifold.manifold_projections.baseclass.ManifoldProjection 
    :members:


SVD projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The singular value decomposition of a matrix :math:`\mathbf{X}\in \mathbb{R}^{n \times m}` is

.. math:: \mathbf{X} =  \mathbf{U}  \mathbf{\Sigma}  \mathbf{V}^\intercal


The :class:`.SvdProjection` class is imported using the following command:

>>> from UQpy.dimension_reduction.SvdProjection import SvdProjection

The signature of the initializer is shown below:

.. autoclass:: UQpy.dimension_reduction.SvdProjection
    :members:

------------------------------------------	
The :class:`.KernelComposition` class is imported using the following command:

>>> from UQpy.dimension_reduction.KernelComposition import KernelComposition

One can use the following command to instantiate the class :class:`.KernelComposition`

.. autoclass:: UQpy.dimension_reduction.KernelComposition
    :members:

