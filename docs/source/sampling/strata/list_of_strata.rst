List of available Strata
^^^^^^^^^^^^^^^^^^^^^^^^

----

Rectangular
~~~~~~~~~~~~~~~~~~
Methods
"""""""
.. autoclass:: UQpy.utilities.strata.RectangularStrata
    :members: stratify, fullfact, plot_2d

Attributes
""""""""""
.. autoattribute:: UQpy.utilities.strata.RectangularStrata.strata_number
.. autoattribute:: UQpy.utilities.strata.RectangularStrata.widths

----

Voronoi
~~~~~~~~~~~~~~~~~~
Methods
"""""""
.. autoclass:: UQpy.utilities.strata.VoronoiStrata
    :members: stratify, compute_voronoi_centroid_volume, add_boundary_points_and_construct_delaunay

Attributes
""""""""""
.. autoattribute:: UQpy.utilities.strata.VoronoiStrata.voronoi
.. autoattribute:: UQpy.utilities.strata.VoronoiStrata.vertices

----

Delaunay
~~~~~~~~~~~~~~~~~~
Methods
"""""""
.. autoclass:: UQpy.utilities.strata.DelaunayStrata
    :members: stratify, compute_delaunay_centroid_volume

Attributes
""""""""""
.. autoattribute:: UQpy.utilities.strata.DelaunayStrata.delaunay
.. autoattribute:: UQpy.utilities.strata.DelaunayStrata.centroids
