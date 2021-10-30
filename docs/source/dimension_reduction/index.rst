DimensionReduction
====================

.. automodule:: UQpy.dimension_reduction

This module contains the classes and methods to perform the point-wise and multi point data-based dimensionality reduction via projection onto the Grassmann manifold and Diffusion Maps, respectively. Further, interpolation in the tangent space centered at a given point on the Grassmann manifold can be performed. In addition, dataset reconstruction and dimension reduction can be performed via the Proper Orthogonal Decomposition method and the Higher-order Singular Value Decomposition for solution snapshots in the form of second-order tensors.

The module ``UQpy.dimension_reduction`` currently contains the following classes:

* ``Grassmann``: Class for for analysis of samples on the Grassmann manifold.

* ``DiffusionMaps``: Class for multi point data-based dimensionality reduction.

* ``POD``: Class for data reconstruction and data dimension reduction.


.. toctree::
   :maxdepth: 2
   :caption: Dimension Reduction