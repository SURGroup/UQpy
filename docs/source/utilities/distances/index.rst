Distances
--------------------------------------

A collection of distances between points in the Euclidean space and between subspaces on the Grassmann manifold.

The Grassmannian (or Riemannian) distance is a metric that assigns nonnegative values to each pair of subspaces
:math:`Y_1, Y_2 \in \mathbb{R}^{n \times p}` on the Grassmann manifold :math:`\mathcal{G}(p, n)`. Formally,
is defined as the length of the shortest geodesic connecting the two points on :math:`\mathcal{G}(p, n)`.
:py:mod:`UQpy` introduces various Grassmann distances derived from the principal angles :cite:`Distances_1`.

.. math:: 0 \leq \theta_1 \leq \theta_2 \leq \ldots \leq \theta_p \leq \pi/2,

where the first principal angle :math:`\theta_1` is the smallest angle between all pairs of unit vectors in the
first and the second subspaces. Practically, the principal angles can be calculated from the singular value
decomposition (SVD) of :math:`\mathbf{Y}_1'\mathbf{Y}_2`,

.. math:: \mathbf{Y}_1'\mathbf{Y}_2 = \mathbf{U}\cos(\Theta)\mathbf{V}'

where :math:`\cos(\Theta)=\text{diag}(\cos\theta_1, \ldots,\cos\theta_p)`. This definition of distance can be extended
to cases where :math:`\mathbf{Y}_1` and :math:`\mathbf{Y}_2` have different number of columns :math:`p`. More
information can be found in :cite:`Distances_2`.

Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Distances` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.Distance import Distance

.. autoclass:: UQpy.utilities.distances.baseclass.Distance
    :members: compute_distance


GrassmannianDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GrassmannianDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.GrassmannianDistance import GrassmannianDistance

The abstract :class:`.GrassmannianDistance` class is a blueprint for all classes in :mod:`.grassmannian_distances` module.
It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.utilities.distances.baseclass.GrassmannianDistance
    :members: calculate_distance_matrix



Asimov
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.AsimovDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.AsimovDistance import AsimovDistance

One can use the following command to instantiate the class :class:`.AsimovDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.AsimovDistance
    :members:


Binet-Cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BinetCauchyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance import BinetCauchyDistance

One can use the following command to instantiate the class :class:`.BinetCauchyDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.BinetCauchyDistance
    :members:


Fubini-Study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.FubiniStudyDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance import FubiniStudyDistance

One can use the following command to instantiate the class :class:`.FubiniStudyDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.FubiniStudyDistance
    :members:



Geodesic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.GeodesicDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.GeodesicDistance import GeodesicDistance

One can use the following command to instantiate the class :class:`.GeodesicDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.GeodesicDistance
    :members:


Martin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MartinDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.MartinDistance import MartinDistance

One can use the following command to instantiate the class :class:`.MartinDistance`


.. autoclass:: UQpy.utilities.distances.grassmannian_distances.MartinDistance
    :members:



Procrustes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProcrustesDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance import ProcrustesDistance

One can use the following command to instantiate the class :class:`.ProcrustesDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProcrustesDistance
    :members:


Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ProjectionDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.ProjectionDistance import ProjectionDistance

One can use the following command to instantiate the class :class:`.ProjectionDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.ProjectionDistance
    :members:


Spectral
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SpectralDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.grassmannian_distances.SpectralDistance import SpectralDistance

One can use the following command to instantiate the class :class:`.SpectralDistance`

.. autoclass:: UQpy.utilities.distances.grassmannian_distances.SpectralDistance
    :members:

------
:TODO: Write details about the specific distances

EuclideanDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The abstract :class:`UQpy.utilities.distances.baseclass.EuclideanDistance` class is a blueprint for classes in :mod:`.euclidean_distances` module.
It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.utilities.distances.baseclass.EuclideanDistance
    :members: calculate_distance_matrix

All the distances classes below are wrappers around the :py:mod:`scipy.spatial.distance` module, written in an
object-oriented fashion to fit the needs of :py:mod:`UQpy.dimension_reduction` module.


Bray-Curtis Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BrayCurtisDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.BrayCurtisDistance import BrayCurtisDistance

One can use the following command to instantiate the class :class:`.BrayCurtisDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.BrayCurtisDistance
    :members:

Canberra Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CanberraDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CanberraDistance import CanberraDistance

One can use the following command to instantiate the class :class:`.CanberraDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CanberraDistance
    :members:

Chebyshev Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ChebyshevDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.ChebyshevDistance import ChebyshevDistance

One can use the following command to instantiate the class :class:`.ChebyshevDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.ChebyshevDistance
    :members:

CityBlock Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CityBlockDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CityBlockDistance import CityBlockDistance

One can use the following command to instantiate the class :class:`.CityBlockDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CityBlockDistance
    :members:

Correlation Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CorrelationDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CorrelationDistance import CorrelationDistance

One can use the following command to instantiate the class :class:`.CorrelationDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CorrelationDistance
    :members:

Cosine Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CosineDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CosineDistance import CosineDistance

One can use the following command to instantiate the class :class:`.CosineDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CosineDistance
    :members:

Dice Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.DiceDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.DiceDistance import DiceDistance

One can use the following command to instantiate the class :class:`.DiceDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.DiceDistance
    :members:

Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.distances.euclidean_distances.EuclideanDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.EuclideanDistance import EuclideanDistance

One can use the following command to instantiate the class :class:`UQpy.utilities.distances.euclidean_distances.EuclideanDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.EuclideanDistance
    :members:

Hamming Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.HammingDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.HammingDistance import HammingDistance

One can use the following command to instantiate the class :class:`.HammingDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.HammingDistance
    :members:

Jaccard Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.JaccardDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.JaccardDistance import JaccardDistance

One can use the following command to instantiate the class :class:`.JaccardDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.JaccardDistance
    :members:

Jensen-Shannon Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.JensenShannonDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.JensenShannonDistance import JensenShannonDistance

One can use the following command to instantiate the class :class:`.JensenShannonDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.JensenShannonDistance
    :members:

Kulczynski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.KulczynskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.KulczynskiDistance import KulczynskiDistance

One can use the following command to instantiate the class :class:`.KulczynskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.KulczynskiDistance
    :members:

Kulsinski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.KulsinskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.KulsinskiDistance import KulsinskiDistance

One can use the following command to instantiate the class :class:`.KulsinskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.KulsinskiDistance
    :members:

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MahalanobisDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MahalanobisDistance import MahalanobisDistance

One can use the following command to instantiate the class :class:`.MahalanobisDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MahalanobisDistance
    :members:

Matching Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MatchingDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MatchingDistance import MatchingDistance

One can use the following command to instantiate the class :class:`.MatchingDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MatchingDistance
    :members:

Minkowski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MinkowskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MinkowskiDistance import MinkowskiDistance

One can use the following command to instantiate the class :class:`.MinkowskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MinkowskiDistance
    :members:

Rogers-Tanimoto Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.RogersTanimotoDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.RogersTanimotoDistance import RogersTanimotoDistance

One can use the following command to instantiate the class :class:`.RogersTanimotoDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.RogersTanimotoDistance
    :members:


Russell-Rao Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.RussellRaoDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.RussellRaoDistance import RussellRaoDistance

One can use the following command to instantiate the class :class:`.RussellRaoDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.RussellRaoDistance
    :members:

Sokal-Michener Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SokalMichenerDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SokalMichenerDistance import SokalMichenerDistance

One can use the following command to instantiate the class :class:`.SokalMichenerDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SokalMichenerDistance
    :members:

Sokal-Sneath Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SokalSneathDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SokalSneathDistance import SokalSneathDistance

One can use the following command to instantiate the class :class:`.SokalSneathDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SokalSneathDistance
    :members:

Squared Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SquaredEuclideanDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SquaredEuclideanDistance import SquaredEuclideanDistance

One can use the following command to instantiate the class :class:`.SquaredEuclideanDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SquaredEuclideanDistance
    :members:

Standardized Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.StandardizedEuclideanDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.StandardizedEuclideanDistance import StandardizedEuclideanDistance

One can use the following command to instantiate the class :class:`.StandardizedEuclideanDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.StandardizedEuclideanDistance
    :members:

Yule Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.YuleDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.YuleDistance import YuleDistance

One can use the following command to instantiate the class :class:`.YuleDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.YuleDistance
    :members: