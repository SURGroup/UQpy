"""

Simplex Sampling
==================================

This example shows the use of the Monte Carlo sampling class. In particular:
"""

#%% md
#
# - How to define the Simplex sampling method supported by UQpy
# - How to draw samples using Simplex sampling
# - How to plot simplices as well as their corresponding samples.

#%%

#%% md
#
# Initially we have to import the necessary modules.

#%%

from UQpy.sampling import SimplexSampling
import numpy as np
import matplotlib.pyplot as plt

#%% md
#
# Define Simplices
# ----------------------------------------------
# Create an array of 3 points in 2-D, which will be the vertex coordinates of the simplex.

#%%

vertex = np.array([[0, 0], [0.5, 1], [1, 0]])

#%% md
#
# Add new samples to existing sampling object
# ----------------------------------------------
# Use Simplex class to generate uniformly distributed samples. This class needs two input parameters,
# i.e. nodes and nsamples. Nodes are the vertex coordinates of simplex and nsamples is the number of new samples to be
# generated. In this example, we are generating ten new samples inside our simplex.

#%%

x = SimplexSampling(nodes=vertex, nsamples=10, random_state=1)

#%% md
#
# Plot the samples of the simplex
# ------------------------------------
# A schematic illustration of the 2-D simplex and new samples generated using Simplex class are presented below.

#%%

plt.plot(np.array([0, 0.5, 1, 0]), np.array([0, 1, 0, 0]), color='blue')
plt.scatter(x.samples[:, 0], x.samples[:, 1], color='red')
plt.show()

#%% md
#
# User can also define a Simplex object using vertices and generate samples using 'run' method.

#%%

y = SimplexSampling(nodes=vertex)
y.run(nsamples=5)

