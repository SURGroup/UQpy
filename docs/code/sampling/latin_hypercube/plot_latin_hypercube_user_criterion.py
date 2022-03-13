"""

User Design Criterion
==================================

This example illustrates use of user-defined latin hypercube design criteria. In particular:
"""

# %% md
#
# - How to define the Monte Carlo sampling method supported by UQpy
# - How to append new samples to the existing ones
# - How to transform existing samples to the unit hypercube
# - How to plot histograms for each one of the sampled parameters and 2D scatter of the produced samples

# %%

# %% md
#
# Initially we have to import the necessary modules.

# %%

from UQpy.sampling import LatinHypercubeSampling
import matplotlib.pyplot as plt
from UQpy.distributions import Uniform
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import *
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Criterion


# %% md
#
# Create user-defined criterion
# ----------------------------------------------
# In order to create a user defined criterion, a concrete implementation of the :class:`.Criterion` abstract class must
# be created. The *generate_samples* method must be implemented, which receives as input the random_state, the randomly
# generated latin hypercube samples.

# %%


class UserCriterion(Criterion):
    def generate_samples(self, random_state):
        lhs_samples = np.zeros_like(self.samples)
        for j in range(self.samples.shape[1]):
            order = np.random.permutation(self.samples.shape[0])
            lhs_samples[:, j] = self.samples[order, j]
        return lhs_samples


# %% md
#
# Define Latin Hypercube sampling
# ----------------------------------------------
# In order to initialize the LatinHypercube sampling class, the user needs to define a list of distributions
# for each one of the parameters that need to be sampled.
#
# Apart from the distributions list, the number of samples  *nsamples* to be drawn is required. The *random_state*
# parameter defines the seed of the random generator.
#
# Finally, the design criterion can be defined by the user. The default case is the :class:`.Random`.
# For more details on the various criteria you can refer to the documentation of the criteria
# :class:`.Random`, :class:`.Centered`, :class:`.Maximin`, :class:`.MinCorrelation`
#
# In the case of user-defined criteria an instantiation of the UserCriterion class is provided instead of the
# built-in criteria.


# %%

dist1 = Uniform(loc=0., scale=1.)
dist2 = Uniform(loc=0., scale=1.)

lhs_user_defined = LatinHypercubeSampling(distributions=[dist1, dist2], nsamples=5,
                                          criterion=UserCriterion())
print(lhs_user_defined._samples)

# %% md
#
# Plot the samples
# ------------------------------------
#
# The samples generated using the LatinHypercube sampling method can be retrieved using the *samples* attribute. This
# attribute is a numpy.ndarray

# %%

# plot the samples
fig, ax = plt.subplots()
plt.title('LHS sampling - User Criterion')
plt.scatter(lhs_user_defined._samples[:, 0], lhs_user_defined._samples[:, 1], marker='o')
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
