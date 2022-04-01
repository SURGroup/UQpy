"""

Delaunay Stratified Sampling
==================================

In this example, the stratified sampling method is employed to generate samples from an exponential distribution
using Delaunay stratification.
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.DelaunayStrata` and :class:`.TrueStratifiedSampling` class from :py:mod:`UQpy.sampling`.

#%%

from UQpy.sampling.stratified_sampling.TrueStratifiedSampling import TrueStratifiedSampling
from UQpy.sampling.stratified_sampling.strata import DelaunayStrata
from UQpy.distributions import Exponential
import numpy as np
import matplotlib.pyplot as plt


#%% md
#
# Run :class:`.TrueStratifiedSampling` for 25 samples.
#
# - 2 dimensions
# - Five strata in each dimension
# - Exponential distribution with location parameter = 1 and scale parameter = 1.
#
# Create a distribution object.

#%%

marginals = [Exponential(loc=1., scale=1.), Exponential(loc=1., scale=1.)]

#%% md
#
# Equal number of samples in each stratum
# ---------------------------------------
#
# Create strata object using :class:`.VoronoiStrata` class.

#%%

seeds = np.array([[0, 0], [0.4, 0.8], [1, 0], [1, 1]])
strata_obj = DelaunayStrata(seeds=seeds)
sts_obj = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj)


#%% md
#
# Figure shows the stratification of domain using randomly generated seed points. Notice that :class:`.DelaunayStrata`
# class include the corners of :math:`[0, 1]^{dimension}` hypercube before constructing Delaunay Triangulation. In this
# plot, orange points are the seed points and left corner is also included in the delaunay construction

#%%

plt.triplot(strata_obj.delaunay.points[:, 0], strata_obj.delaunay.points[:, 1], strata_obj.delaunay.simplices)
plt.plot(seeds[:, 0], seeds[:, 1], 'or')
plt.show()

#%% md
#
# Run stratified sampling

#%%

sts_obj = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj)
sts_obj.run(nsamples_per_stratum=2)

#%% md
#
# Plot the resulting stratified samples and the boundaries of the strata in the :math:`U(0,1)` space.

#%%

plt.triplot(strata_obj.delaunay.points[:, 0], strata_obj.delaunay.points[:, 1], strata_obj.delaunay.simplices)
plt.plot(seeds[:, 0], seeds[:, 1], 'or')
plt.plot(sts_obj.samplesU01[:, 0], sts_obj.samplesU01[:, 1], 'dm')
plt.title('Stratified Sample - U(0,1)')
plt.show()

sts_obj.weights


#%% md
#
# Proportional Sampling
# ----------------------
# Delaunay class can generate samples proportional to volume of each stratum.

#%%

sts_obj = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj)
sts_obj.run(nsamples=10)


#%% md
#
# It can be noticed that new sample in each stratum is proportional to volume

#%%

print('Volume: ', sts_obj.strata_object.volume)
print('Number of samples in each stratum: ', sts_obj.nsamples_per_stratum)

plt.triplot(strata_obj.delaunay.points[:, 0], strata_obj.delaunay.points[:, 1], strata_obj.delaunay.simplices)
plt.plot(seeds[:, 0], seeds[:, 1], 'or')
plt.plot(sts_obj.samplesU01[:, 0], sts_obj.samplesU01[:, 1], 'dm')
plt.show()