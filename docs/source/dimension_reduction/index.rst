Dimension Reduction
====================

.. automodule:: UQpy.dimension_reduction

This module contains various classes and methods to perform dimensionality reduction in :py:mod:`UQpy`. The module is
structured around the "manifold assumption", i.e., that high dimensional data lie close to a lower-dimension manifold.
:py:mod:`.dimension_reduction` offers point-wise dimensionality reduction of a dataset via projection on the Grassmann
manifold. This module provides a collection of Grassmann distances and kernels that encode the proximity between points
on the Grassmann manifold. This module also provides an efficent implementation of  Diffusion maps method that aim to
find a low-dimensional embedding of this manifold, and the Proper Orthogonal Decomposition (POD) and  Higher-order
Singular Value Decomposition (HOSVD) methods, that can be used for data reconstruction and dimension reduction for
solution snapshots in the form of second-order tensors.


.. toctree::
   :maxdepth: 1
   :caption: Current capabilities

    Grassmann manifold <grassmann/index>
    Diffusion maps <dmaps>
    HOSVD <hosvd>
    POD <pod>
