"""

S-Curve
=====================

This example shows how to use the :class:`.DiffusionMaps` class to reveal the embedded structure of S-Curve data.
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.utilities.kernels.GaussianKernel import GaussianKernel
from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps
from sklearn.datasets import make_swiss_roll, make_s_curve

# %% md
#
# Sample points randomly following a parametric curve and plot the 3D graphic.

# %%

n = 4000  # number of samples

# generate point cloud
X, X_color = make_s_curve(n, random_state=3, noise=0)

# %% md
# plot the point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X_color, cmap=plt.cm.Spectral,
           )
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("S-curve data")
ax.view_init(10, 70)

# %% md
#
# Case 1: Find the optimal parameter of the Gaussian kernel scale epsilon
kernel = GaussianKernel()

dmaps_object = DiffusionMaps.build_from_data(data=X,
                                             alpha=1.0, n_eigenvectors=9,
                                             is_sparse=True, neighbors_number=100,
                                             # epsilon=0.0018,
                                             kernel=kernel)

print('epsilon', kernel.epsilon)

# %% md
# Fit the data and calculate the embedding, the eigenvectors and eigenvalues
dmaps_object.fit()

# %% md
#
# Find the parsimonious representation of the eigenvectors. Identify the two most informative
# eigenvectors.

index, residuals = DiffusionMaps.parsimonious(dmaps_object.eigenvectors, 2)

print('most informative eigenvectors:', index)

# %% md
#
# Plot the diffusion coordinates

DiffusionMaps._plot_eigen_pairs(dmaps_object.eigenvectors, pair_indices=index, color=X_color, figure_size=[12, 12])
plt.show()
