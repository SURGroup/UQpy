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

The City Block (Manhattan) distance between two 1D arrays, `x` and `y`, is given by:

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

The Correlation distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = 1 - \dfrac{(x-\bar{x})\cdot(y-\bar{y})}{||x-\bar{x}||_2||y-\bar{y}||_2}

where :math:`\bar{x}` denotes the mean of the elements of :math:`x` and :math:`x\cdot y` denotes the dot product.

The :class:`.CorrelationDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CorrelationDistance import CorrelationDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the class :class:`.CorrelationDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CorrelationDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CorrelationDistance.distance_matrix

Cosine Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Cosine distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = 1 - \dfrac{x\cdot y}{||x||_2||y||_2}

where :math:`x\cdot y` denotes the dot product.

The :class:`.CosineDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.CosineDistance import CosineDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the class :class:`.CosineDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.CosineDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.CosineDistance.distance_matrix


L2 Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`UQpy.utilities.distances.euclidean_distances.L2Distance` class is imported using the following command:
The L2 distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = ||x - y||_2

The :class:`UQpy.utilities.distances.euclidean_distances.L2Distance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.L2Distance import L2Distance


Methods
~~~~~~~~~~

One can use the following command to instantiate the class :class:`UQpy.utilities.distances.euclidean_distances.L2Distance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.L2Distance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.L2Distance.distance_matrix


Minkowski Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Minkowski distance between two 1D arrays, `x` and `y`, is given by:

.. math:: d(x,y) = ||x - y||_p = \left(\sum_i |x_i-y_i|^p \right)^{1/p}.

The :class:`.MinkowskiDistance` class is imported using the following command:

>>> from UQpy.utilities.distances.euclidean_distances.MinkowskiDistance import MinkowskiDistance

Methods
~~~~~~~~~~

One can use the following command to instantiate the class :class:`.MinkowskiDistance`

.. autoclass:: UQpy.utilities.distances.euclidean_distances.MinkowskiDistance
    :members:

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.utilities.distances.euclidean_distances.MinkowskiDistance.distance_matrix

