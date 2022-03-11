Diffusion Maps
--------------------------------

Diffusion Maps (:cite:t:`COIFMAN20065`) is a nonlinear dimension reduction technique used to learn (i.e., parametrize) a manifold from some data.
Diffusion maps are based on the assumption that the data is represented in a high-dimensional space, while the points lie
close to a low-dimensional manifold. The algorithm constructs a Markov Chain
based on the available data. The probabilities of this Markov Chain define how probable a transition between two  points is
in one time step of the diffusion process. Then, the eigenfunctions of the Markov matrix are used to obtain a
coordinate system that reveals the embedded geometry of the data.



The :class:`.DiffusionMaps` class is imported using the following command:

>>> from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps

One can use the following method to instantiate the class :class:`.DiffusionMaps`

DiffusionMaps Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
~~~~~~~~~~~
.. autoclass:: UQpy.dimension_reduction.diffusion_maps.DiffusionMaps
    :members: build_from_data, fit, parsimonious, estimate_cut_off, estimate_epsilon

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


