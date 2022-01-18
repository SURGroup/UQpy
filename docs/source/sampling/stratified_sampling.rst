Stratified Sampling
---------------------

Stratified sampling is a variance reduction technique that divides the parameter space into a set of disjoint and space-filling strata. Samples are then drawn from these strata in order to improve the space-filling properties of the sample design. Stratified sampling allows for unequally weighted samples, such that a Monte Carlo estimator of the quantity :math:`E[Y]` takes the following form:

.. math:: E[Y] \approx \sum_{i=1}^N w_i Y_i

where :math:`w_i` are the sample weights and :math:`Y_i` are the model evaluations. The individual sample weights are computed as:

.. math:: w_i = \dfrac{V_{i}}{N_{i}}

where :math:`V_{i}\le 1` is the volume of stratum :math:`i` in the unit hypercube (i.e. the probability that a random sample will fall in stratum :math:`i`) and :math:`N_{i}` is the number of samples drawn from stratum :math:`i`.


:py:mod:`UQpy` supports several stratified sampling variations that vary from conventional stratified sampling designs
to advanced gradient informed methods for adaptive stratified sampling. These class structures facilitate a highly flexible and varied range of stratified
sampling designs that can be extended in a straightforward way. Specifically, the existing classes allow stratification
of n-dimensional parameter spaces based on three common spatial discretizations: a rectilinear decomposition into
hyper-rectangles (orthotopes), a Voronoi decomposition, and a Delaunay decomposition. This structure is based on three classes:

1. The :class:`.Strata` class defines the geometric structure of the stratification of the parameter space and it has
three existing subclasses - :class:`.Rectangular`, :class:`.Voronoi`, and :class:`.Delaunay` that correspond to
geometric decompositions of the parameter space based on rectilinear strata of orthotopes, strata composed of Voronoi
cells, and strata composed of Delaunay simplexes respectively. These classes live in the :py:mod:`UQpy.sampling.stratified_sampling.strata` folder.

.. toctree::
   :maxdepth: 1

    Strata Base Class <strata/strata_class>
    Rectangular Strata <strata/rectangular_strata>
    Delaunay Strata <strata/delaunay_strata>
    Voronoi Strata <strata/voronoi_strata>
    Adding a new Strata <strata/adding_new_strata>

2. The :class:`.TrueStratifiedSampling` class defines a set of subclasses used to draw samples from strata defined by a :class:`.Strata` class object.

3. The :class:`.RefinedStratifiedSampling` class defines a set of subclasses for refinement of :class:`.TrueStratifiedSampling` stratified sampling designs.



StratifiedSampling Class
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.TrueStratifiedSampling` class is the parent class for stratified sampling. The various
:class:`.TrueStratifiedSampling` classes generate random samples from a specified probability distribution(s) using
stratified sampling with strata specified by an object of one of the :class:`.Strata` classes.

Methods
""""""""""""""""""""""""
.. autoclass:: UQpy.sampling.stratified_sampling.TrueStratifiedSampling
    :members: transform_samples, run

Attributes
"""""""""""
.. autoattribute:: UQpy.sampling.stratified_sampling.TrueStratifiedSampling.weights
.. autoattribute:: UQpy.sampling.stratified_sampling.TrueStratifiedSampling.samples
.. autoattribute:: UQpy.sampling.stratified_sampling.TrueStratifiedSampling.samplesU01


