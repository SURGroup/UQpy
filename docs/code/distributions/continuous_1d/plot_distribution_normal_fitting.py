"""

Distribution fitting
==================================

This examples showcases the calculation of a distributions parameters based on fitting available data
"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

from UQpy.distributions.collection.Normal import Normal

#%% md
#
# Define a Normal distribution and use the fit method.
# ------------------------------------------------------
#
# Parameters to be learnt should be instantiated as None.
# Note that the fit method of each distribution returns a dictionary containing as keys the names of parameters
# and its values are their data fitted values.

#%%

normal1 = Normal(loc=None, scale=None)
fitted_parameters1 = normal1.fit(data=[-4, 2, 2, 1])
print(fitted_parameters1)

normal2 = Normal(loc=0., scale=None)
fitted_parameters2 = normal2.fit(data=[-4, 2, 2, 1])
print(fitted_parameters2)

