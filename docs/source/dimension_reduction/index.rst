Dimension Reduction
====================

.. automodule:: UQpy.dimension_reduction

This module contains various classes and methods to perform dimensionality reduction in :py:mod:`UQpy`. The module is
structured around the "manifold assumption", which states that high dimensional data are assumed to lie on or close to a
lower-dimension manifold. The :py:mod:`.dimension_reduction` module offers both point-wise and multi-point
dimensionality reduction methods.
For point-wise (or single point) dimension reduction, high-dimensional data points are projected onto the Grassmann
manifold using the :class:`.GrassmannProjection` class. The :class:`.GrassmannOperations` class is then used to perform
operations on the Grassmann manifold and the :class:`.GrassmannInterpolation` class can be used to interpolate on the
manifold.

The :py:mod:`.dimension_reduction` module has three additional multi-point dimension reduction methods including linear
and nonlinear methods. For linear dimension reduction, the Proper Orthogonal Decomposition (:class:`.POD`) and
Higher-order Singular Value Decomposition (:class:`.HigherOrderSVD`) classes are available for dimension reduction of
vector/matrix-valued quantities and tensor-valued quantities, respectively. The :class:`.POD` baseclass contains
subclasses for the Direct POD (:class:`.DirectPOD`) and the Snapshot POD (:class:`.SnapshotPOD`). For nonlinear
dimension provides an efficent implementation of Diffusion maps (:class:`.DiffusionMaps`).


.. toctree::
   :maxdepth: 2
   :caption: Current capabilities

    Grassmann manifold <grassmann/index>
    Diffusion maps <dmaps>
    POD <pod>
    HOSVD <hosvd>

