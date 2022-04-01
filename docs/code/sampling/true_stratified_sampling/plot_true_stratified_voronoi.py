"""

Voronoi Stratified Sampling
==================================

In this example, the stratified sampling method is employed to generate samples from an exponential distribution using
Voronoi stratification.
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.TrueStratifiedSampling` class from :py:mode:`UQpy.sampling`.

#%%

from UQpy.sampling.stratified_sampling.TrueStratifiedSampling import TrueStratifiedSampling
from UQpy.sampling.stratified_sampling.strata import VoronoiStrata
from UQpy.distributions import Exponential
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

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
# ----------------------------------------
# Create strata object using :class:`.VoronoiStrata` class.

#%%

strata_obj = VoronoiStrata(seeds_number=8, dimension=2)
sts_vor_obj = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj,  random_state=3)

#%% md
#
# Figure shows the stratification of domain using randomly generated seed points

#%%

strata_obj

voronoi_plot_2d(strata_obj.voronoi)
plt.title('Stratified Sample - U(0,1)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

#%% md
#
# Run stratified sampling

#%%

sts_vor_obj.run(nsamples_per_stratum=3)

#%% md
#
# Plot the resulting stratified samples and the boundaries of the strata in the :math:`U(0,1)` space.

#%%

voronoi_plot_2d(strata_obj.voronoi)
plt.title('Stratified Sample - U(0,1)')
plt.plot(sts_vor_obj.samplesU01[:, 0], sts_vor_obj.samplesU01[:, 1], 'dm')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

sts_vor_obj.weights

#%% md
#
# Proportional Sampling
# ---------------------
# :class:`.TrueStratifiedSampling` class can generate samples proportional to volume of each stratum.

#%%

sts_vor_obj1 = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj, random_state=1)
sts_vor_obj1.run(nsamples=10)

#%% md
#
# It can be noticed that new sample in each stratum is proportional to volume

#%%

print('Volume: ', sts_vor_obj1.strata_object.volume)
print('Number of samples in each stratum: ', sts_vor_obj1.nsamples_per_stratum)


voronoi_plot_2d(strata_obj.voronoi)
plt.title('Stratified Sample - U(0,1)')
plt.plot(sts_vor_obj1.samplesU01[:, 0], sts_vor_obj1.samplesU01[:, 1], 'dm')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()