
Refined Stratified Sampling
-----------------------------

Refined Stratified Sampling (RSS) is a sequential sampling procedure that adaptively refines the stratification of the
parameter space to add samples. There are four variations of RSS currently available in :py:mod:`UQpy`. First, the procedure
works with either rectangular stratification (i.e. using :class:`.Rectangular`) or Voronoi stratification
(i.e. using :class:`.Voronoi`). For each of these, two refinement procedures are available. The first is a randomized
algorithm where strata are selected at random according to their probability weight. This algorithm is described
in [10]_. The second is a gradient-enhanced version (so-called GE-RSS) that draws samples in stata that possess both
large probability weight and have high variance. This algorithm is described in [11]_.

Refined Stratified Sampling Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All variations of Refined Stratifed Sampling are implemented in the :class:`.RefinedStratifiedSampling` class.

Extension of the RSS class for new algorithms can be accomplished by adding new a new strata that implements the
appropriate methods.


RefinedStratifiedSampling Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parent class for Refined Stratified Sampling [10]_, [11]_.
This is the parent class for all refined stratified sampling methods. This parent class only provides the
framework for refined stratified sampling and cannot be used directly for the sampling. Sampling is done by
calling the child class for the desired algorithm.

.. autoclass:: UQpy.sampling.RefinedStratifiedSampling
    :members:

.. [10] M.D. Shields, K. Teferra, A. Hapij, and R.P. Daddazio, "Refined Stratified Sampling for efficient Monte Carlo based uncertainty quantification," Reliability Engineering and System Safety,vol.142, pp.310-325,2015.
.. [11] M.D. Shields, "Adaptive Monte Carlo analysis for strongly nonlinear stochastic systems." Reliability Engineering & System Safety 175 (2018): 207-224.