Distances
--------------------------------------

A collection of distances between points in the Euclidean space and between subspaces on the Grassmann manifold.

The Grassmannian (or Riemannian) distance is a metric that assigns nonnegative values to each pair of subspaces :math:`Y_1, Y_2 \in \mathbb{R}^{n \times p}` on the Grassmann manifold :math:`\mathcal{G}(p, n)`. Formally, is defined as the length of the shortest geodesic connecting the two points on :math:`\mathcal{G}(p, n)`. :py:mod:`UQpy` introduces various Grassmann distances derived from the principal angles [1]_

.. math:: 0 \leq \theta_1 \leq \theta_2 \leq \ldots \leq \theta_p \leq \pi/2,

where the first principal angle :math:`\theta_1` is the smallest angle between all pairs of unit vectors in the first and the second subspaces. Practically, the principal angles can be calculated from the singular value decomposition (SVD) of :math:`\mathbf{Y}_1'\mathbf{Y}_2`,

.. math:: \mathbf{Y}_1'\mathbf{Y}_2 = \mathbf{U}\cos(\Theta)\mathbf{V}'

where :math:`\cos(\Theta)=\text{diag}(\cos\theta_1, \ldots,\cos\theta_p)`. This definition of distance can be extended to cases where :math:`\mathbf{Y}_1` and :math:`\mathbf{Y}_2` have different number of columns :math:`p`. More information can be found in [2]_.

RiemannianDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The abstract :class:`.RiemannianDistance` class is a blueprint for classes in :mod:`.distances` module. It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.dimension_reduction.RiemannianDistance
    :members:



Asimov
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.AsimovDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.AsimovDistance import AsimovDistance

One can use the following command to instantiate the class :class:`.AsimovDistance`

.. autoclass:: UQpy.dimension_reduction.distances.AsimovDistance
    :members:


Binet-Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BinetCauchyDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.BinetCauchyDistance import BinetCauchyDistance

One can use the following command to instantiate the class :class:`.BinetCauchyDistance`

.. autoclass:: UQpy.dimension_reduction.distances.BinetCauchyDistance
    :members:


Fubini-Study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.FubiniStudyDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.FubiniStudyDistance import FubiniStudyDistance

One can use the following command to instantiate the class :class:`.FubiniStudyDistance`

.. autoclass:: UQpy.dimension_reduction.distances.FubiniStudyDistance
    :members:



Geodesic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GeodesicDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.GeodesicDistance import GeodesicDistance

One can use the following command to instantiate the class :class:`.GeodesicDistance`

.. autoclass:: UQpy.dimension_reduction.distances.GeodesicDistance
    :members:


Martin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MartinDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.MartinDistance import MartinDistance

One can use the following command to instantiate the class :class:`.MartinDistance`

.. autoclass:: UQpy.dimension_reduction.distances.MartinDistance
    :members:



Procrustes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProcrustesDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.ProcrustesDistance import ProcrustesDistance

One can use the following command to instantiate the class :class:`.ProcrustesDistance`

.. autoclass:: UQpy.dimension_reduction.distances.ProcrustesDistance
    :members:


Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProjectionDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.ProjectionDistance import ProjectionDistance

One can use the following command to instantiate the class :class:`.ProjectionDistance`

.. autoclass:: UQpy.dimension_reduction.distances.ProjectionDistance
    :members:


Spectral
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SpectralDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.SpectralDistance import SpectralDistance

One can use the following command to instantiate the class :class:`.SpectralDistance`

.. autoclass:: UQpy.dimension_reduction.distances.SpectralDistance
    :members:



Euclidean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.EuclideanDistance` class is imported using the following command:

>>> from UQpy.dimension_reduction.distances.EuclideanDistance import EuclideanDistance

One can use the following command to instantiate the class :class:`.EuclideanDistance`

.. autoclass:: UQpy.dimension_reduction.distances.EuclideanDistance
    :members:

.. [1] G. H. Golub and C.F.V. Loan. Matrix computations (3rd ed.). Baltimore, MD, USA: Johns Hopkins University Press, 1996.

.. [2]  K. Ye and L.H. Lim, Schubert varieties and distances between subspaces of different dimensions, SIAM Journal on Matrix Analysis and Applications, 2016, 37, 1176–1197.
