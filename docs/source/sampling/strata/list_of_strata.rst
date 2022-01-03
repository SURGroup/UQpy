List of available Strata
^^^^^^^^^^^^^^^^^^^^^^^^

----

Rectangular
~~~~~~~~~~~~~~~~~~
Methods
"""""""
.. autoclass:: UQpy.sampling.stratified_sampling.strata.RectangularStrata
    :members: stratify, fullfact, plot_2d

Attributes
""""""""""
.. autoattribute:: UQpy.utilities.sampling.stratified_sampling.RectangularStrata.strata_number
.. autoattribute:: UQpy.utilities.sampling.stratified_sampling.RectangularStrata.widths

----

Voronoi
~~~~~~~~~~~~~~~~~~
Methods
"""""""
.. autoclass:: UQpy.sampling.stratified_sampling.strata.VoronoiStrata
    :members: stratify, compute_voronoi_centroid_volume, add_boundary_points_and_construct_delaunay

Attributes
""""""""""
.. autoattribute:: UQpy.sampling.stratified_sampling.strata.VoronoiStrata.voronoi
.. autoattribute:: UQpy.sampling.stratified_sampling.strata.VoronoiStrata.vertices

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
