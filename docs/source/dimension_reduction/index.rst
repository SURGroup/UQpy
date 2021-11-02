Dimension Reduction
====================

.. automodule:: UQpy.dimension_reduction

This module contains the classes and methods to perform the point-wise and multi point data-based dimensionality reduction via projection onto the Grassmann manifold and Diffusion Maps, respectively. Further, interpolation in the tangent space centered at a given point on the Grassmann manifold can be performed. In addition, dataset reconstruction and dimension reduction can be performed via the Proper Orthogonal Decomposition method and the Higher-order Singular Value Decomposition for solution snapshots in the form of second-order tensors.

The module :py:mod:`.dimension_reduction` currently contains the following classes:

* :class:`.Grassmann`: Class for for analysis of samples on the Grassmann manifold.

* :class:`.DiffusionMaps`: Class for multi point data-based dimensionality reduction.

* :class:`.POD`: Class for data reconstruction and data dimension reduction.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Dimension Reduction

    POD <pod>
    HOSVD <hosvd>
    Diffusion Maps <diffusion_maps>
    Grassmann <grassmann>
    Distances <distances/index>
    Kernels <kernels/index>
    Manifold Projections <manifold_projections/index>