Diffusion Maps
--------------------------------

Diffusion Maps (:cite:t:`COIFMAN20065`) is a nonlinear dimension reduction technique used to learn (i.e., parametrize)
a manifold from some data. Diffusion maps are based on the assumption that the data is represented in a high-dimensional
space, while the points lie on or close to a low-dimensional manifold. The algorithm operates by defining a graph over
the data. On this graph a random walk is defined with a Markov transition probability determined by a distance between
data points. An eigendecomposition of the Markov transition probability matrix is used to obtain lower-dimensional
coordinates that reveal the instrinsic structure of the data.

DiffusionMaps Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.DiffusionMaps` class is imported using the following command:

>>> from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps

One can use the following method to instantiate the :class:`.DiffusionMaps` class.



Methods
~~~~~~~~~~~
.. autoclass:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps
    :members: build_from_data, fit, parsimonious, estimate_epsilon

Attributes
~~~~~~~~~~~~~
.. autoattribute:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps.transition_matrix
.. autoattribute:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps.diffusion_coordinates
.. autoattribute:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps.eigenvectors
.. autoattribute:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps.eigenvalues

Examples
~~~~~~~~~~~~~

.. toctree::

   Diffusion Maps Examples <../auto_examples/dimension_reduction/diffusion_maps/index>


