"""
This module contains functionality for all the sampling methods supported in ``UQpy``.

The module currently contains the following classes:

- ``MCS``: Class to perform Monte Carlo sampling.
- ``LHS``: Class to perform Latin hypercube sampling.
- ``mcmc``: Class to perform Markov Chain Monte Carlo sampling.
- ``IS``: Class to perform Importance sampling.
- ``AKMCS``: Class to perform adaptive kriging Monte Carlo sampling.
- ``StratifiedSampling``: Class to perform stratified sampling.
- ``refined_stratified_sampling``: Class to perform refined stratified sampling.
- ``strata``: Class to perform stratification of the unit hypercube.
- ``Simplex``: Class to uniformly sample from a simplex.
"""

from UQpy.sampling.mcmc import *
from UQpy.sampling.refined_stratified_sampling import *

from UQpy.sampling.AdaptiveKriging import AdaptiveKriging
from UQpy.sampling.ImportanceSampling import ImportanceSampling
from UQpy.sampling.LatinHypercubeSampling import LatinHypercubeSampling
from UQpy.sampling.MonteCarloSampling import MonteCarloSampling
from UQpy.sampling.SimplexSampling import SimplexSampling
from UQpy.sampling.StratifiedSampling import StratifiedSampling
