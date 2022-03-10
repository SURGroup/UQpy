"""

Monte Carlo Sampling
==================================

This example shows the use of the Monte Carlo sampling class. In particular:
"""

#%% md
#
# - How to define the Monte Carlo sampling method supported by UQpy
# - How to append new samples to the existing ones
# - How to transform existing samples to the unit hypercube
# - How to plot histograms for each one of the sampled parameters and 2D scatter of the produced samples

#%%

#%% md
#
# Initially we have to import the necessary modules.

#%%

from UQpy.sampling import MonteCarloSampling
from UQpy.distributions.collection.Normal import Normal
import numpy as np
import matplotlib.pyplot as plt

#%% md
#
# Define Monte Carlo sampling
# ----------------------------------------------
# In order to define a Monte Carlo sampling object, the user has to define a distribution for each one of the parameters
# that need sampling. These distributions form a list that is provided to initializer and is the only necessary
# argument. Two optional arguments also exist and are the *nsamples* and the a *random_state*. *nsamples* corresponds to
# the number of samples that must be drawn using the sampling method. If *nsamples* is defined at the initializer,
# sampling is automatically performed, otherwise no samples are drawn. *random_state* is a seed used as a starting point
# for the random number generator.


#%%

dist1 = Normal(loc=0., scale=1.)
dist2 = Normal(loc=0., scale=1.)

x = MonteCarloSampling(distributions=[dist1, dist2], nsamples=5,
                       random_state=np.random.RandomState(123))

#%% md
#
# Add new samples to existing sampling object
# ----------------------------------------------
# Since we have already sampled 5 points by defining *nsamples* at the initializer, we can append new samples to the
# monte carlo object by using the run method. This method draws and appends samples to existing ones, while allowing the
# *random_state* to change.
#
# Note that all samples drawn do not belong to the unit hypercube space. Even though the *samplesU01* attributes exists,
# it will initially be None. To transform the samples and at the same time populate the attribute, the user need to call
# the *transform_u01()* function.

#%%

print(x.samples)

x.run(nsamples=2, random_state=np.random.RandomState(23))
print(x.samples)

x.transform_u01()
print(x.samplesU01)

#%% md
#
# Plot the samples
# ------------------------------------
#
# The samples generated using the MonteCarlo method can be retrieved using the *samples* attribute. Ths attribute is
# a numpy.ndarray.

#%%

# plot the samples
fig, ax = plt.subplots()
plt.title('MC sampling')
plt.scatter(x.samples[:, 0], x.samples[:, 1], marker='o')
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()

fig, ax = plt.subplots()
plt.title('Histogram:parameter #1')
plt.hist(x.samples[:, 0])
ax.yaxis.grid(True)
ax.xaxis.grid(True)

fig, ax = plt.subplots()
plt.title('Histogram:parameter #2')
plt.hist(x.samples[:, 1])
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()

#%% md
#
# Use the MCS.run method in order to add 2 samples in the existing ones.

#%%

x.run(nsamples=2)
print()
print('MCS extended samples:')
print(x.samples)

# Use the MCS.transform_u01 method in order transform the samples to the Uniform [0,1] space.
x.transform_u01()
print()
print('MCS transformed samples:')
print(x.samplesU01)

#%% md
#
# We are going to run MCS for a multivariate normal distribution random variables.

#%%

from UQpy.distributions import MultivariateNormal
dist = MultivariateNormal(mean=[1., 2.], cov=[[4., -0.9], [-0.9, 1.]])

x1 = MonteCarloSampling(distributions=dist, nsamples=5, random_state=np.random.RandomState(456))
print()
print(x1.samples)

# plot the samples
fig, ax = plt.subplots()
plt.title('MC sampling')
plt.scatter(x1.samples[:, 0], x1.samples[:, 1], marker='o')
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()

# Use the MCS.run method in order to add 2 samples in the existing ones.
x1.run(nsamples=2)
print()
print('MCS extended samples:', x1.samples)

#%% md
#
# Use the MCS.transform_u01 method in order transform the samples to the Uniform [0,1] space.
# However, in order to use this method all distributions need to have a ``cdf`` method.
# For example, in this case the MVNormal distribution does not have a ``cdf`` method so the code
# will give an error.

#%%

x1.transform_u01()


#%% md
#
# We are going to run MCS for a multivariate normal distribution and a standard normal distribution.

#%%

x2 = MonteCarloSampling(distributions=[dist1, dist], nsamples=5, random_state=np.random.RandomState(789))
print()
print('MCS samples:', x2.samples)

print(len(x2.samples))
print(len(x2.samples[0]))

# Extract samples for the multivariate distribution
samples = np.zeros(shape=(len(x2.samples), len(x2.samples[1])))
for i in range(len(x2.samples)):
    samples[i, :] = x2.samples[i][1]
print()
print('MVNormal samples:', samples)

# Use the MCS.run method in order to add 2 samples in the existing ones.
x2.run(nsamples=2)
print()
print('MCS extended samples:', x2.samples)

#%% md
#
# Draw samples from a continuous and discrete distribution

#%%

from UQpy.distributions import Binomial
dist3 = Binomial(n=5, p=0.4)
x3 = MonteCarloSampling(distributions=[dist1, dist3], nsamples=5)

print()
print('MCS samples:', x3.samples)