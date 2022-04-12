"""

User-defined Rosenbrock distribution
=====================================

This examples shows the use of the multivariate normal distributions class. In particular:
"""


import numpy as np
from UQpy.distributions import DistributionND
import matplotlib.pyplot as plt

#%% md
#
# Example with a custom distribution
# ----------------------------------
# In order to define a new distribution, the user must extend the one of the abstract base classes
# :class:`.DistributionContinuous1D`, :class:`.DistributionDiscrete1D` or :class:`.DistributionND`.
# For the purpose of this example a
# new multivariate Rosenbrock distribution is defined. Note that three methods are implemented, namely, the
# :code:`__init__`
# which allows the user to define custom distribution arguments, as well as the pdf and log_pdf of this new
# distribution. Note that it is required for the user to call the :code:`__init__` method of the baseclass by providing
# all arguments names and values required in the custom distribution initializer. In our case, the parameter name
# :code:`p` and its values are provided to the :code:`super().__init__` method.

#%%


class Rosenbrock(DistributionND):
    def __init__(self, p=20.):
        super().__init__(p=p)

    def pdf(self, x):
        return np.exp(-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / self.parameters['p'])

    def log_pdf(self, x):
        return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / self.parameters['p']

#%% md
#
# Initialize custom distribution
# ----------------------------------
# Given the newly defined distribution class, a Rosenbrock distribution object can be defined by providing to the
# initializer the user-defined argument :code:`p`. Since the Rosenbrock distribution extends the
# :class:`.DistributionND` class, methods
# and attributes of the baseclass are already available. For instance, the user can retrieve the already defined
# parameters using the :code:`parameters` attribute and subsequently update them using the :code:`update_parameters`
# method.

#%%

dist = Rosenbrock(p=20)
print(dist.parameters)
dist.update_parameters(p=40)
print(dist.parameters)

dist = Rosenbrock(p=20)
print(dist.parameters)

#%% md
#
# Plot pdf of the user defined distribution
# ------------------------------------------
#

#%%

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(-5, 8, 0.1)
y = np.arange(-5, 50, 0.1)
X, Y = np.meshgrid(x, y)
Z = dist.pdf(x=np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1))
CS = ax.contour(X, Y, Z.reshape(X.shape))
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Contour plot of custom pdf - Rosenbrock')
plt.show()

#%% md
#
# Check if the user-defined distribution has rvs, pdf and update_parameters method
# ----------------------------------------------------------------------------------
#

#%%

print('Does the rosenbrock distribution have an rvs method?')
print(hasattr(dist, 'rvs'))

print('Does the rosenbrock distribution have an pdf method?')
print(hasattr(dist, 'pdf'))

print('Does the rosenbrock distribution have an update_parameters method?')
print(hasattr(dist, 'update_parameters'))