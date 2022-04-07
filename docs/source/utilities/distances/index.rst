Distances
--------------------------------------

A collection of distance measures between points in the Euclidean space and between subspaces on the Grassmann manifold.

Each distance function in :py:mod:`UQpy` is defined as a subclass of the :class:`.Distance` class. The
:class:`.Distance` class has two further subclasses for Euclidean distances (:class:`.baseclass.EuclideanDistance,`) and
Grassmann distances (:class:`.GrassmannianDistance`). Individual dsitances, depending on their type, are defined as
subclasses of these.

Distance Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Distances` class is imported using the following command:

>>> from UQpy.utilities.distances.baseclass.Distance import Distance

.. autoclass:: UQpy.utilities.distances.baseclass.Distance
    :members:

Types of Distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Euclidean Distances <euclidean_distances>
    Grassmannian Distances <grassmann_distances>