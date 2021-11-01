Stratified Sampling
---------------------

Stratified sampling is a variance reduction technique that divides the parameter space into a set of disjoint and space-filling strata. Samples are then drawn from these strata in order to improve the space-filling properties of the sample design. Stratified sampling allows for unequally weighted samples, such that a Monte Carlo estimator of the quantity :math:`E[Y]` takes the following form:

.. math:: E[Y] \approx \sum_{i=1}^N w_i Y_i

where :math:`w_i` are the sample weights and :math:`Y_i` are the model evaluations. The individual sample weights are computed as:

.. math:: w_i = \dfrac{V_{i}}{N_{i}}

where :math:`V_{i}\le 1` is the volume of stratum :math:`i` in the unit hypercube (i.e. the probability that a random sample will fall in stratum :math:`i`) and :math:`N_{i}` is the number of samples drawn from stratum :math:`i`.


:py:mod:`UQpy` supports several stratified sampling variations that vary from conventional stratified sampling designs
to advanced gradient informed methods for adaptive stratified sampling. Stratified sampling capabilities are built in
:py:mod:`UQpy` from three sets of classes. These class structures facilitate a highly flexible and varied range of stratified
sampling designs that can be extended in a straightforward way. Specifically, the existing classes allow stratification
of n-dimensional parameter spaces based on three common spatial discretizations: a rectilinear decomposition into
hyper-rectangles (orthotopes), a Voronoi decomposition, and a Delaunay decomposition. The three parent classes are:

1. The :class:`.Strata` class defines the geometric structure of the stratification of the parameter space and it has three existing subclasses - :class:`.Rectangular`, :class:`.Voronoi`, and:class:`.Delaunay` that correspond to geometric decompositions of the parameter space based on rectilinear strata of orthotopes, strata composed of Voronoi cells, and strata composed of Delaunay simplexes respectively. These classes live in the :py:mod:`UQpy.utilities.strata` folder.

2. The :class:`.StratifiedSampling` class defines a set of subclasses used to draw samples from strata defined by a :class:`.Strata` class object.

3. The :class:`.RefinedStratifiedSampling` class defines a set of subclasses for refinement of :class:`.StratifiedSampling` stratified sampling designs.

Strata Class
^^^^^^^^^^^^^

The :class:`.Strata` class is the parent class that defines the geometric decomposition of the parameter space. All geometric decompositions in the :class:`.Strata` class are performed on the `n`-dimensional unit :math:`[0, 1]^n` hypercube. Specific stratifications are performed by subclassing the :class:`.Strata` class. There are currently three stratifications available in the :class:`.Strata` class, defined through the subclasses :class:`.Rectangular`, :class:`.Voronoi`, and :class:`.Delaunay`.


Strata Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.utilities.strata.baseclass.Strata
    :members:

.. autoclass:: UQpy.utilities.strata.Rectangular
    :members:

.. autoclass:: UQpy.utilities.strata.Voronoi
    :members:

.. autoclass:: UQpy.utilities.strata.Delaunay
    :members:

Adding a new :class:`.Strata` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding a new type of stratification requires creating a new subclass of the :class:`.Strata` class that defines the
desired geometric decomposition. This subclass must have a :meth:`stratify` method that overwrites the corresponding
method in the parent class and performs the stratification.


StratifiedSampling Class
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.StratifiedSampling` class is the parent class for stratified sampling. The various
:class:`.StratifiedSampling` classes generate random samples from a specified probability distribution(s) using
stratified sampling with strata specified by an object of one of the :class:`.Strata` classes.

StratifiedSampling Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.sampling.StratifiedSampling
    :members:

New Stratified Sampling Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extension of the stratified sampling capabilities in :py:mod:`UQpy` can be performed through subclassing from the three main classes.
First, the user can define a new geometric decomposition of the parameter space by creating a new subclass of the :class:`Strata` class.
To implement a new stratified sampling method based on a new stratification, the user must write a new subclass of the :class:`.Strata` class defining the new decomposition.
The details of these subclasses and their requirements are outlined in the sections discussing the respective classes.

.. [9] K. Tocher. "The art of simulation." The English Universities Press, London, UK; 1963.

