"""

Grassmannian diffusion maps
=====================

This example shows how to use the UQpy DiffusionMaps class for points on the Grassmann
manifold. Reference [gdmaps]
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.dimension_reduction import DiffusionMaps, ProjectionKernel, SvdProjection

#%% md
#
# Sample points randomly following a parametric curve and plot the 3D graphic.

#%%

n = 4000  # number of samples


# instantiate a diffusion maps object.
# Use a Gaussian kernel
kernel = ProjectionKernel()

# ------------------------------------------------------------------------------------------
# Case 1: Find the optimal parameter of the Gaussian kernel scale epsilon

gdmaps_object = DiffusionMaps(alpha=1.0, eigenvectors_number=9,
                              kernel=kernel)


# Fit the data and calculate the embedding, the eigenvectors and eigenvalues
diff_coords, eigenvalues, eigenvectors = gdmaps_object.fit(X)

#%% md
#
# Plot the Grassmannian diffusion coordinates

figure_params = dict(figsize=[10, 10])
n_eigenvectors = eigenvectors.shape[1] - 1

f, ax = plt.subplots(
    nrows=int(np.ceil(n_eigenvectors / 2)), ncols=2, sharex=True, sharey=True, **figure_params)
enum = 0

for i, idx in enumerate(range(n_eigenvectors + 1)):
    if i == 1:
        enum = 1
    else:
        i = i - enum

    _ax = ax[i // 2, i - (i // 2) * 2]
    _ax.scatter( diff_coords[:, 1], diff_coords[:, idx], cmap=plt.cm.Spectral, c=X_color)

    _ax.set_title(r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(1, idx))

#%% md
#
# Find the parsimonious representation of the eigenvectors. Identify the two most informative
# eigenvectors.

index, residuals = DiffusionMaps.parsimonious(eigenvectors, 2)

print('most informative eigenvectors:', index)

plt.show()
