Sample Methods
==============

This module contains functionality for all the sampling methods supported in :py:mod:`UQpy`.

The module currently contains the following classes:

- :class:`.MonteCarloSampling`: Class generating random samples from a specified probability distribution(s).

- :class:`.LatinHypercubeSampling`: Class generating random samples from a specified probability distribution(s) using Latin hypercube sampling.

- :class:`.StratifiedSampling`: Class is a variance reduction technique that divides the parameter space into a set of disjoint and space-filling strata

- :class:`.RefinedStratifiedSampling`: Class is a sequential sampling procedure that adaptively refines the stratification of the parameter space to add samples

- :class:`.SimplexSampling`: Class generating uniformly distributed samples inside a simplex.

- :class:`.AdaptiveKriging`: Class generating samples adaptively using a specified Kriging-based learning function in a general Adaptive Kriging-Monte Carlo Sampling (AKMCS) framework

- :class:`.MCMC`: The goal of Markov Chain Monte Carlo is to draw samples from some probability distribution which is hard to compute

- :class:`.ImportanceSampling`: Importance sampling (IS) is based on the idea of sampling from an alternate distribution and reweighing the samples to be representative of the target distribution

.. toctree::
   :hidden:
   :maxdepth: 1

    Monte Carlo Sampling <monte_carlo>
    Latin Hypercube Sampling <latin_hypercube>
    Stratified Sampling <stratified_sampling>
    Refined Stratified Sampling <refined_stratified_sampling>
    Simplex Sampling <simplex>
    Adaptive Kriging <akmcs>
    Markov Chain Monte Carlo <mcmc/index>
    Importance Sampling <importance_sampling>





