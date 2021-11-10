Diffusion Maps
--------------------------------

Diffusion Maps [1]_ is a nonlinear dimension reduction technique used to learn (i.e., parametrize) a manifold from some data.
Diffusion maps are based on the assumption that the data is represented in a high-dimensional space, while the points lie
close to a low-dimensional manifold. The algorithm constructs a Markov Chain
based on the available data. The probabilities of this Markov Chain define how probable a transition between two  points is
in one time step of the diffusion process. Then, the eigenfunctions of the Markov matrix are used to obtain a
coordinate system that reveals the embedded geometry of the data.



The :class:`.DiffusionMaps` class is imported using the following command:

>>> from UQpy.dimension_reduction.dmaps.DiffusionMaps import DiffusionMaps

One can use the following command to instantiate the class :class:`.DiffusionMaps`

.. autoclass:: UQpy.dimension_reduction.dmaps.DiffusionMaps
    :members:

.. [1] R. R. Coifman, S. Lafon. Diffusion maps. Applied Computational Harmonic Analysis, 2006, 21(1), p.5–30.
