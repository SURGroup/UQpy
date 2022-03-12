"""

Distribution Discrete 1D example
=================================

This examples shows the use of the univariate discrete distributions class. In particular:
"""
#%% md
#
# - How to define one of the univariate discrete distributions supported by UQpy
# - How to extract the moments of the distribution
# - How to draw random samples from the distribution

#%%

#%% md
#
# Import the necessary modules.

#%%

import matplotlib.pyplot as plt
from UQpy.distributions.collection.Binomial import Binomial

#%% md
#
# Example of a 1D discrete distribution
# --------------------------------------
#
# Define a univariate binomial distribution.
# By using the __bases__ attribute we can verify that the Binomial distribution extends the DistributionDiscrete1D
# baseclass, while in order to define the Binomial distribution, two parameters are required, namely, *n* and *p*.

#%%

print(Binomial.__bases__)
dist = Binomial(n=5, p=0.4)


#%% md
#
# Print the mean, standard deviation, skewness, and kurtosis of the distribution.
# -------------------------------------------------------------------------------
# Using the moments method existing in all univariate distributions, the user can retrieve the available
# moments. The order in which the moments are extracted can be seen in the moments_list variable.

#%%

moments_list = ['mean', 'variance', 'skewness', 'kurtosis']
m = dist.moments()
print('Moments with inherited parameters:')
for i, moment in enumerate(moments_list):
    print(moment + ' = {0:.2f}'.format(m[i]))

# %% md
#
# Generate 5000 random samples from the binomial distribution.
# -------------------------------------------------------------
#
# The number of samples is provided as nsamples (default 1).
# The user can fix the seed of the pseudo random generator via input random_state.
#
# Important: the output of rvs is a (nsamples, 1) ndarray.

# %%

y1 = dist.rvs(nsamples=5000)
print('Shape of output provided by rvs is (nsamples, dimension), i.e. here:')
print(y1.shape)
plt.hist(y1[:, 0], bins=50)
plt.xlabel('x')
plt.ylabel('count')
plt.show()






