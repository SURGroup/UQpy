"""

Swiss role
=====================

This example shows how to use the :class:`.DiffusionMaps` class to reveal the embedded structure of noisy Swiss role data.
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.dimension_reduction import DiffusionMaps, GaussianKernel
from sklearn.datasets import make_swiss_roll

#%% md
#
# Sample points randomly following a parametric curve and plot the 3D graphic.

#%%

n = 4000  # number of samples

# generate point cloud
X, X_color = make_swiss_roll(n, random_state=3, noise=0)


#%% md
# plot the point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X_color, cmap=plt.cm.Spectral,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Swiss role data")
ax.view_init(10, 70)


#%% md
# ------------------------------------------------------------------------------------------
# Case 1: Find the optimal parameter of the Gaussian kernel scale epsilon
kernel = GaussianKernel()

dmaps_object = DiffusionMaps.create_from_data(data=X, alpha=1.0, eigenvectors_number=9,
                                              optimize_parameters=True,
                                              kernel=kernel)


print('epsilon', kernel.epsilon)


#%% md
# Fit the data and calculate the embedding, the eigenvectors and eigenvalues
diff_coords, eigenvalues, eigenvectors = dmaps_object.fit()


#%% md
#
# Find the parsimonious representation of the eigenvectors. Identify the two most informative
# eigenvectors.

index, residuals = DiffusionMaps.parsimonious(eigenvectors, 2)

print('most informative eigenvectors:', index)

#%% md

# Plot the diffusion coordinates

DiffusionMaps._plot_eigen_pairs(eigenvectors, pair_indices=[1, 2], color=X_color, figure_size=[12, 12])
plt.show()

