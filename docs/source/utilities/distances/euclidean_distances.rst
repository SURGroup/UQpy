Euclidean Distances
--------------------------------------


EuclideanDistance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The abstract :class:`UQpy.utilities.distances.baseclass.EuclideanDistance` class is a blueprint for classes in :mod:`.euclidean_distances` module.
It allows the user to define a set of methods that must be created within any child classes built from this abstract class.

.. autoclass:: UQpy.utilities.distances.baseclass.EuclideanDistance
    :members: calculate_distance_matrix

List of Available Distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the distances classes below are subclasses of the :class:`.EuclideanDistance` class.


Bray-Curtis Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Bray-Curtis distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = \dfrac{\sum_i |x_i - y_i|}{\sum_i |x_i + y_i|}

The :class:`.BrayCurtisDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.BrayCurtisDistance import BrayCurtisDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the :class:`.BrayCurtisDistance` class.

.. autoclass:: UQpy.utilities.distances.euclidean_distances.BrayCurtisDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.BrayCurtisDistance.distance_matrix

Canberra Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Canberra distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = \sum_i \dfrac{|x_i - y_i|}{|x_i| + |y_i|}

The :class:`.CanberraDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CanberraDistance import CanberraDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the :class:`.CanberraDistance` class.

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CanberraDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.CanberraDistance.distance_matrix

Chebyshev Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Chebyshev distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = \max_i |x_i-y_i|

The :class:`.ChebyshevDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.ChebyshevDistance import ChebyshevDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the :class:`.ChebyshevDistance` class:

.. autoclass:: UQpy.utilities.distances.euclidean_distances.ChebyshevDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.ChebyshevDistance.distance_matrix

CityBlock Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Canberra distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = \sum_i |x_i - y_i|

The :class:`.CityBlockDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CityBlockDistance import CityBlockDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the :class:`.CityBlockDistance` class

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CityBlockDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.CityBlockDistance.distance_matrix

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

.. autoclass:: UQpy.utilities.distances.euclidean_distances.KulsinksiDistance
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