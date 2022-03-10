"""

Rectangular Stratified Sampling
==================================

In this example, the stratified sampling method is employed to generate samples from an exponential distribution.
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the STS class from UQpy.SampleMethods.

#%%

from UQpy.sampling.stratified_sampling.TrueStratifiedSampling import TrueStratifiedSampling
from UQpy.distributions import Exponential
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from UQpy.sampling.stratified_sampling.strata import RectangularStrata


#%% md
#
# Run STS for 25 samples.
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
# Create strata with equal volume
# --------------------------------
# Create a strata object using RectangularStrata class.

#%%

strata = RectangularStrata(strata_number=[5, 5])

#%% md
#
# Generate samples using RectangularSTS class, one sample is generate inside each stratum.

#%%

x_sts = TrueStratifiedSampling(distributions=marginals,
                               strata_object=strata,
                               nsamples_per_stratum=1)


#%% md
#
# Plot the resulting stratified samples and the boundaries of the strata in the U(0,1) space.

#%%

fig = strata.plot_2d()
plt.title('Stratified Sample - U(0,1)')
plt.scatter(x_sts.samplesU01[:, 0], x_sts.samplesU01[:, 1], color='r')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()


print(x_sts.weights)

#%% md
#
# Plot the resulting stratified exponential samples and the boundaries of the strata in the exponential space.

#%%

fig, ax = plt.subplots()
plt.title('Stratified Sample - Exponential')
plt.scatter(x_sts.samples[:, 0], x_sts.samples[:, 1])
ax.set_yticks([1.0, expon.ppf(0.2, 1, 1), expon.ppf(0.4, 1, 1), expon.ppf(0.6, 1, 1), expon.ppf(0.8, 1, 1),
               expon.ppf(0.99, 1, 1)])
ax.set_xticks([1.0, expon.ppf(0.2, 1, 1), expon.ppf(0.4, 1, 1), expon.ppf(0.6, 1, 1), expon.ppf(0.8, 1, 1),
               expon.ppf(0.99, 1, 1)])
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.ylim(1, expon.ppf(0.99, 1, 1))
plt.xlim(1, expon.ppf(0.99, 1, 1))
plt.show()

print(x_sts.samples)


#%% md
#
# Create stratification using seeds and widths
# --------------------------------
# Strata object can be initiated by defining seeds and widths of the strata.

#%%

seeds = np.array([[0, 0], [0.4, 0], [0, 0.5], [0.4, 0.5]])
widths = np.array([[0.4, 0.5], [0.6, 0.5], [0.4, 0.5], [0.6, 0.5]])
strata_obj = RectangularStrata(seeds=seeds, widths=widths)


#%% md
#
# Generate samples using RectangularSTS class. User can control the number of samples generated inside each stratum.
# In this illustration, 10 samples are generated such that nsamples_per_stratum governs the number of sa

#%%

sts_obj = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj, random_state=20)
sts_obj.run(nsamples_per_stratum=[1, 2, 3, 4])

#%% md
#
# Plot show the strata and samples generated in each stratum.

#%%

fig = strata_obj.plot_2d()
plt.title('Stratified Sample - U(0,1)')
plt.scatter(sts_obj.samplesU01[:, 0], sts_obj.samplesU01[:, 1], color='r')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

#%% md
#
# Probability weights corresponding to each samples computed using Stratified sampling.

#%%

sts_obj.weights

#%% md
#
# Create stratification using input file
# --------------------------------
# Strata object can be defined using a input file, which contains the seeds and widths of each stratum.

#%%

strata_obj1 = RectangularStrata(input_file='strata.txt')

#%% md
#
# Generate samples inside eaach stratum using RectangularSTS class.

#%%

sts_obj1 = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj1, nsamples_per_stratum=1)

fig = strata_obj1.plot_2d()
plt.title('Stratified Sample - U(0,1)')
plt.scatter(sts_obj1.samplesU01[:, 0], sts_obj1.samplesU01[:, 1], color='r')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

#%% md
#
# Proportional sampling
# ----------------------
# RectangularSTS class can generate samples proportional to volume of each stratum.

#%%

strata_obj.random_state = 24
sts_obj2 = TrueStratifiedSampling(distributions=marginals, strata_object=strata_obj)
sts_obj2.run(nsamples=10)

#%% md
#
# It can be noticed that new sample in each stratum is proportional to volume.

#%%

print('Volume: ', sts_obj2.strata_object.volume)
print('Number of samples in each stratum: ', sts_obj2.nsamples_per_stratum)


fig = strata_obj.plot_2d()
plt.title('Stratified Sample - U(0,1)')
plt.scatter(sts_obj.samplesU01[:, 0], sts_obj.samplesU01[:, 1], color='r')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
