Strata Class
^^^^^^^^^^^^^

The :class:`.Strata` class is the parent class that defines the geometric decomposition of the parameter space. All geometric decompositions in the :class:`.Strata` class are performed on the `n`-dimensional unit :math:`[0, 1]^n` hypercube. Specific stratifications are performed by subclassing the :class:`.Strata` class. There are currently three stratifications available in the :class:`.Strata` class, defined through the subclasses :class:`.Rectangular`, :class:`.Voronoi`, and :class:`.Delaunay`.


Methods
"""""""
.. autoclass:: UQpy.sampling.stratified_sampling.strata.baseclass.Strata
    :members: stratify

Attributes
""""""""""""""
.. autoattribute:: UQpy.sampling.stratified_sampling.strata.baseclass.Strata.seeds
.. autoattribute:: UQpy.sampling.stratified_sampling.strata.baseclass.Strata.volume