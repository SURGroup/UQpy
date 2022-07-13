
Refined Stratified Sampling
-----------------------------

Refined Stratified Sampling (RSS) is a sequential sampling procedure that adaptively refines the stratification of the
parameter space to add samples. There are four variations of RSS currently available in :py:mod:`UQpy`. First, the procedure
works with either rectangular stratification (i.e. using :class:`.Rectangular`) or Voronoi stratification
(i.e. using :class:`.Voronoi`). For each of these, two refinement procedures are available. The first is a randomized
algorithm where strata are selected at random according to their probability weight. This algorithm is described
in :cite:`Rss1`. The second is a gradient-enhanced version (so-called GE-RSS) that draws samples in strata that possess both
large probability weight and have high variance. This algorithm is described in :cite:`Rss2`.

Refined Stratified Sampling Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All variations of Refined Stratifed Sampling are implemented in the :class:`.RefinedStratifiedSampling` class.
This class provides the framework for refined stratified sampling. with the aid of an underlying stratification generated
in a previous sampling of type :class:`TrueStratifiedSampling`, as well as a :class:`.Refinement` algorithm an adaptive
sampling and refinement of the stratification can be performed.

The :class:`.RefinedStratifiedSampling` class is imported using the following command:

>>> from UQpy.sampling.stratified_sampling.RefinedStratifiedSampling import RefinedStratifiedSampling

Methods
""""""""""""""""""""""""
.. autoclass:: UQpy.sampling.RefinedStratifiedSampling
    :members: run

Attributes
""""""""""""""""""""""""
.. autoattribute:: UQpy.sampling.RefinedStratifiedSampling.samples
.. autoattribute:: UQpy.sampling.RefinedStratifiedSampling.samplesU01

Examples
""""""""""""""""""""""""

.. toctree::

   Refined Stratified Sampling Examples <../auto_examples/sampling/refined_stratified_sampling/index>


Stratification Refinement Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.RefinedStratifiedSampling` allows for an adaptive refinement of existing strata. This adaptive refinement
procedure can be performed based on different algorithms. Each algorithm provides a different approach on selecting and
refining the available :class:`.Strata`, which can be either randomly, or based on advanced selection techniques.
In order to accommodate all possible refinement procedures, the :class:`.Refinement` baseclass is created. The user only
needs to implement the :py:meth:`update_samples` method, thus allowing the implementation of different adaptive strata
refinement techniques.

.. autoclass:: UQpy.sampling.stratified_sampling.refinement.baseclass.Refinement
    :members: update_samples

The :class:`.RandomRefinement` class is imported using the following command:

>>> from UQpy.sampling.stratified_sampling.refinement.RandomRefinement import RandomRefinement

.. autoclass:: UQpy.sampling.stratified_sampling.refinement.RandomRefinement

The :class:`.GradientEnhancedRefinement` class is imported using the following command:

>>> from UQpy.sampling.stratified_sampling.refinement.GradientEnhancedRefinement import GradientEnhancedRefinement

.. autoclass:: UQpy.sampling.stratified_sampling.refinement.GradientEnhancedRefinement
