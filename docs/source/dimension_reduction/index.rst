Dimension Reduction
====================

.. automodule:: UQpy.dimension_reduction

This module contains various classes and methods to perform dimensionality reduction in :py:mod:`UQpy`. The module is
structured around the "manifold assumption", which states that high dimensional data are assumed to lie on or close to a lower-dimension manifold.
The :py:mod:`.dimension_reduction` module offers both point-wise and multi-point dimensionality reduction methods.
For point-wise, or single point, dimension reduction The of a dataset via projection onto the Grassmann
manifold.  This module also provides an efficent implementation of  Diffusion maps method that aim to
find a low-dimensional embedding of this manifold, and the Proper Orthogonal Decomposition (POD) and  Higher-order
Singular Value Decomposition (HOSVD) methods, that can be used for data reconstruction and dimension reduction for
solution snapshots in the form of second-order tensors.


.. toctree::
   :maxdepth: 2
   :caption: Current capabilities

    Grassmann manifold <grassmann/index>
    Diffusion maps <dmaps>
    HOSVD <hosvd>
    POD <pod>
