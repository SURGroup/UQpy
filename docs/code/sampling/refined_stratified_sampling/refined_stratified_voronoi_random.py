"""

Voronoi Refined Stratified Sampling - Random Refinement
============================================================

In this example, Stratified sampling is used to generate samples from Uniform probability distribution and sample are
added using adaptive approach Refined Stratified Sampling.
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.TrueStratifiedSampling` and :class:`.RefinedStratifiedSampling` class from UQpy.

#%%

from UQpy.sampling.stratified_sampling.strata import VoronoiStrata
from UQpy.sampling import TrueStratifiedSampling, RefinedStratifiedSampling, RandomRefinement
from UQpy.distributions import Uniform
import matplotlib.pyplot as plt

#%% md
#
# Create a distribution object.

#%%

marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]

strata = VoronoiStrata(seeds_number=16, dimension=2)

#%% md
#
# Run stratified sampling.

#%%
x = TrueStratifiedSampling(distributions=marginals, strata_object=strata, nsamples_per_stratum=1, random_state=1)

#%% md
#
# Plot the resulting stratified samples and the boundaries of the strata in the :math:`U(0,1)` space.

#%%

from scipy.spatial import voronoi_plot_2d
fig = voronoi_plot_2d(strata.voronoi)
plt.title('Stratified Samples (U(0,1)) - Voronoi Stratification')
plt.plot(x.samples[:, 0], x.samples[:, 1], 'dm')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

#%% md
#
# Using UQpy :class:`.RefinedStratifiedSampling` class to expand samples generated by :class:`.TrueStratifiedSampling`
# class. In this example, two new samples are generated inside cells with maximum weight associated with it.

#%%

refinement = RandomRefinement(strata=strata)
y = RefinedStratifiedSampling(stratified_sampling=x, nsamples=18, refinement_algorithm=refinement,
                              samples_per_iteration=2, random_state=2)

#%% md
#
# The figure shows the voronoi tesselation of initial samples and samples generated using
# :class:`.RefinedStratifiedSampling` class. The :class:`.RefinedStratifiedSampling` class
# creates a Delaunay triangulation using existing samples and generate a new sample inside the triangle with maximum
# volume.

#%%

fig2 = voronoi_plot_2d(strata.voronoi)
plt.title('Refined Stratified Samples - Voronoi Stratification')
plt.plot(x.samples[:16, 0], x.samples[:16, 1], 'dm')
plt.scatter(y.samplesU01[16:18, 0], y.samplesU01[16:18, 1], marker='s', color='black')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

#%% md
#
# Further, :class:`.RefinedStratifiedSampling` class is used to adaptively increase the sample size. In this example,
# samples are randomly added in cell with maximum weights associated with it and new sample generated using
# :class:`.SimplexSampling` class.

#%%

y.run(nsamples=50)

#%% md
#
# In the figure shown below, all samples generated from :class:`.TrueStratifiedSampling` and
# :class:`.RefinedStratifiedSampling` class are plotted.

#%%

fig2 = voronoi_plot_2d(y.refinement_algorithm.strata.voronoi)
plt.title('Refined Stratified Samples - Voronoi Stratification')
plt.plot(x.samples[:16, 0], x.samples[:16, 1], 'dm')
plt.scatter(y.samplesU01[16:, 0], y.samplesU01[16:, 1], marker='s', color='black')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()