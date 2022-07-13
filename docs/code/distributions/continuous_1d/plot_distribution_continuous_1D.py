"""

Distribution Continuous 1D Example
==================================

This examples shows the use of a univariate continuous distributions class. In particular:
"""

#%% md
#
# - How to define one of the univariate distributions supported by UQpy
# - How to plot the pdf and log_pdf of the distribution
# - How to modify the parameters of the distribution
# - How to extract the moments of the distribution
# - How to draw random samples from the distribution

#%%

#%% md
#
# Initially we have to import the necessary modules.

#%%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.distributions import Lognormal

#%% md
#
# Example of a univariate lognormal distribution
# ----------------------------------------------
# In order to define the lognormal distribution, the user must provide its three parameters, *s*, *loc* and *scale*.
# Printing the dist object created after the Lognormal distribution initialization, will display as output the class
# of the distribution as can be seen below. The provided parameters of the distribution can be retrieved using the
# *parameters* attribute available to all distributions.


#%%

dist = Lognormal(s=1., loc=0., scale=np.exp(5))
print(dist)
print(dist.parameters)

#%% md
#
# Plot the pdf of the distribution.
# ------------------------------------
#
# The user must provide x as a ndarray of shape (nsamples, 1) or (nsamples,) - the former if preferred.
# The result of pdf or log_pdf will be a 1D array (nsamples, ).

#%%

x = np.linspace(0.01, 1000, 1000).reshape((-1, 1))  # Use reshape to provide a 2D array (1000, 1)
fig, ax = plt.subplots(ncols=2, figsize=(15, 4))
ax[0].plot(x, dist.pdf(x))  # Do not give params
ax[0].set_xlabel('x')
ax[0].set_ylabel('pdf(x)')
ax[0].set_title('pdf of lognormal distribution')

ax[1].plot(x, dist.log_pdf(x))
ax[1].set_xlabel('x')
ax[1].set_ylabel('log pdf(x)')
ax[1].set_title('Log pdf of lognormal distribution')
plt.show()

print('size of input x:')
print(x.shape)
print('size of dist.pdf(x):')
print(dist.pdf(x).shape)

#%% md
#
# Modify one of the parameters of the distribution.
# -------------------------------------------------
#
# Use the *update_parameters* method. The user must provide as input to the *update_parameters* method the name,
# as well as the updated value of the specified parameter. Note that in case the user provides a non-existing parameter
# name, the *update_parameters* method will raise an exception.

#%%

dist.update_parameters(loc=100.)
print(dist.parameters)

#%% md
#
# Plot the pdf and log_pdf functions of the lognormal distribution with the updated parameters.

#%%

x = np.linspace(0.01, 1000, 1000).reshape((-1, 1))  # Use reshape to provide a 2D array (1000, 1)
fig, ax = plt.subplots(ncols=2, figsize=(15, 4))
ax[0].plot(x, dist.pdf(x))  # Do not give params
ax[0].set_xlabel('x')
ax[0].set_ylabel('pdf(x)')
ax[0].set_title('pdf of lognormal distribution')

ax[1].plot(x, dist.log_pdf(x))
ax[1].set_xlabel('x')
ax[1].set_ylabel('log pdf(x)')
ax[1].set_title('Log pdf of lognormal distribution')
plt.show()

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
# Generate 5000 random samples from the lognormal distribution.
# -------------------------------------------------------------
#
# The number of samples is provided as nsamples (default 1).
# The user can fix the seed of the pseudo-random generator via input random_state.
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