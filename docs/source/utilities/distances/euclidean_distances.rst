Euclidean Distances
--------------------------------------


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

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.BrayCurtisDistance.distance_matrix

Canberra Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CanberraDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CanberraDistance import CanberraDistance

One can use the following command to instantiate the class :class:`.CanberraDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CanberraDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CanberraDistance.distance_matrix

Chebyshev Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.ChebyshevDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.ChebyshevDistance import ChebyshevDistance

One can use the following command to instantiate the class :class:`.ChebyshevDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.ChebyshevDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.ChebyshevDistance.distance_matrix

CityBlock Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CityBlockDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CityBlockDistance import CityBlockDistance

One can use the following command to instantiate the class :class:`.CityBlockDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CityBlockDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CityBlockDistance.distance_matrix

Correlation Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CorrelationDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CorrelationDistance import CorrelationDistance

One can use the following command to instantiate the class :class:`.CorrelationDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CorrelationDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CorrelationDistance.distance_matrix

Cosine Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.CosineDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CosineDistance import CosineDistance

One can use the following command to instantiate the class :class:`.CosineDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CosineDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CosineDistance.distance_matrix

Dice Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.DiceDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.DiceDistance import DiceDistance

One can use the following command to instantiate the class :class:`.DiceDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.DiceDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.DiceDistance.distance_matrix

L2 Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.distances.euclidean_distances.L2Distance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.L2Distance import L2Distance

One can use the following command to instantiate the class :class:`UQpy.utilities.distances.euclidean_distances.L2Distance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.L2Distance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.L2Distance.distance_matrix

Hamming Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.HammingDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.HammingDistance import HammingDistance

One can use the following command to instantiate the class :class:`.HammingDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.HammingDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.HammingDistance.distance_matrix

Jaccard Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.JaccardDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.JaccardDistance import JaccardDistance

One can use the following command to instantiate the class :class:`.JaccardDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.JaccardDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.JaccardDistance.distance_matrix

Jensen-Shannon Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.JensenShannonDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.JensenShannonDistance import JensenShannonDistance

One can use the following command to instantiate the class :class:`.JensenShannonDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.JensenShannonDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.JensenShannonDistance.distance_matrix

Kulczynski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.KulczynskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.KulczynskiDistance import KulczynskiDistance

One can use the following command to instantiate the class :class:`.KulczynskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.KulczynskiDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.KulczynskiDistance.distance_matrix

Kulsinski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.KulsinskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.KulsinskiDistance import KulsinskiDistance

One can use the following command to instantiate the class :class:`.KulsinskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.KulsinksiDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.KulsinksiDistance.distance_matrix

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MahalanobisDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MahalanobisDistance import MahalanobisDistance

One can use the following command to instantiate the class :class:`.MahalanobisDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MahalanobisDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.MahalanobisDistance.distance_matrix

Matching Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MatchingDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MatchingDistance import MatchingDistance

One can use the following command to instantiate the class :class:`.MatchingDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MatchingDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.MatchingDistance.distance_matrix

Minkowski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.MinkowskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MinkowskiDistance import MinkowskiDistance

One can use the following command to instantiate the class :class:`.MinkowskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MinkowskiDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.MinkowskiDistance.distance_matrix

Rogers-Tanimoto Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.RogersTanimotoDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.RogersTanimotoDistance import RogersTanimotoDistance

One can use the following command to instantiate the class :class:`.RogersTanimotoDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.RogersTanimotoDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.RogersTanimotoDistance.distance_matrix


Russell-Rao Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.RussellRaoDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.RussellRaoDistance import RussellRaoDistance

One can use the following command to instantiate the class :class:`.RussellRaoDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.RussellRaoDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.RussellRaoDistance.distance_matrix

Sokal-Michener Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SokalMichenerDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SokalMichenerDistance import SokalMichenerDistance

One can use the following command to instantiate the class :class:`.SokalMichenerDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SokalMichenerDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.SokalMichenerDistance.distance_matrix

Sokal-Sneath Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SokalSneathDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SokalSneathDistance import SokalSneathDistance

One can use the following command to instantiate the class :class:`.SokalSneathDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SokalSneathDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.SokalSneathDistance.distance_matrix

Squared Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SquaredEuclideanDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.SquaredEuclideanDistance import SquaredEuclideanDistance

One can use the following command to instantiate the class :class:`.SquaredEuclideanDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.SquaredEuclideanDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.SquaredEuclideanDistance.distance_matrix

Standardized Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.StandardizedEuclideanDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.StandardizedEuclideanDistance import StandardizedEuclideanDistance

One can use the following command to instantiate the class :class:`.StandardizedEuclideanDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.StandardizedEuclideanDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.StandardizedEuclideanDistance.distance_matrix

Yule Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.YuleDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.YuleDistance import YuleDistance

One can use the following command to instantiate the class :class:`.YuleDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.YuleDistance
    :members:

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.YuleDistance.distance_matrix