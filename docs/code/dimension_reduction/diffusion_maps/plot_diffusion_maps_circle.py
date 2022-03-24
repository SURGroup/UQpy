"""

Circle
===================================================================

This example shows how to use the UQpy DiffusionMaps class to reveal the embedded structure of noisy data.
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the DiffusionMaps class from UQpy implemented in the DimensionReduction module.

# %%

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from UQpy.utilities.kernels.GaussianKernel import GaussianKernel
from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps

# %% md
#
# Sample points randomly following a parametric curve and plot the 3D graphic.

# %%

a = 6
b = 1
k = 10
u = np.linspace(0, 2 * np.pi, 1000)

v = k * u

x0 = (a + b * np.cos(0.8 * v)) * (np.cos(u))
y0 = (a + b * np.cos(0.8 * v)) * (np.sin(u))
z0 = b * np.sin(0.8 * v)

rox = 0.2
roy = 0.2
roz = 0.2
x = x0 + rox * np.random.normal(0, 1, len(x0))
y = y0 + roy * np.random.normal(0, 1, len(y0))
z = z0 + roz * np.random.normal(0, 1, len(z0))

X = np.array([x, y, z]).transpose()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c='b', cmap=plt.cm.Spectral, s=8)
ax.plot(x0, y0, z0, 'r', label='parametric curve')
plt.show()

# %% md
#
# Instantiate the class `DiffusionMaps` using `alpha=1`; `n_evecs=3`, because the first eigenvector is non-informative.
# Moreover, a Gaussian is used in the kernel construction.

# %%

dmaps = DiffusionMaps.build_from_data(data=X, alpha=1, n_eigenvectors=3,
                                      kernel=GaussianKernel(), epsilon=0.3)

# %% md
#
# Use the method `mapping` to compute the diffusion coordinates assuming `epsilon=0.3`.

# %%

dmaps.fit()

# %% md
#
# Plot the second and third diffusion coordinates to reveal the embedded structure of the data.

# %%

color = dmaps.eigenvectors[:, 1]
plt.scatter(dmaps.diffusion_coordinates[:, 1], dmaps.diffusion_coordinates[:, 2], c=color, cmap=plt.cm.Spectral, s=8)
plt.axis('equal')
plt.show()

# %% md
#
# Use the colormap to observe how the embedded structure is distributed in the original set.

# %%

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c=color, cmap=plt.cm.Spectral, s=8)
plt.show()

