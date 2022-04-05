"""

Compute distances between points
==================================

This example shows how to use the UQpy Grassmann class to compute distances
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the Grassmann class from UQpy implemented in the DimensionReduction module.

# %%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UQpy import SvdProjection
import sys

from UQpy.utilities import GrassmannPoint
from UQpy.utilities.distances.baseclass.GrassmannianDistance import GrassmannianDistance
from UQpy.utilities.distances.grassmannian_distances.GeodesicDistance import GeodesicDistance

from UQpy.dimension_reduction import GrassmannOperations

# %% md
#
# Generate four random matrices with reduced rank corresponding to the different samples. The samples are stored in
# `matrices`.

# %%

D1 = 6
r0 = 2  # rank sample 0
r1 = 3  # rank sample 1
r2 = 4  # rank sample 2
r3 = 3  # rank sample 2

np.random.seed(1111)  # For reproducibility.
# Solutions: original space.
Sol0 = np.dot(np.random.rand(D1, r0), np.random.rand(r0, D1))
Sol1 = np.dot(np.random.rand(D1, r1), np.random.rand(r1, D1))
Sol2 = np.dot(np.random.rand(D1, r2), np.random.rand(r2, D1))
Sol3 = np.dot(np.random.rand(D1, r3), np.random.rand(r3, D1))

# Creating a list of solutions.
matrices = [Sol0, Sol1, Sol2, Sol3]

# Plot the solutions
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Matrix 0')
ax1.imshow(Sol0)
ax2.title.set_text('Matrix 1')
ax2.imshow(Sol1)
ax3.title.set_text('Matrix 2')
ax3.imshow(Sol2)
ax4.title.set_text('Matrix 3')
ax4.imshow(Sol3)
plt.show()

# %% md
#
# Instantiate the SvdProjection class that projects the raw data to the manifold.

# %%

manifold_projection = SvdProjection(matrices, p="max")

# %% md
#
# Compute the pairwise distances for :math:`\Psi` and :math:`\Phi`, the left and right -singular eigenvectors,
# respectively, of singular value decomposition of each solution.

# %%
p_dim = [manifold_projection.p] * len(manifold_projection.u)
pairwise_distance = GeodesicDistance().calculate_distance_matrix(points=manifold_projection.u,
                                                                 p_dim=p_dim)
print(pairwise_distance)

# %% md
#
# Compute the distance between 2 points.

# %%

distance01 = GeodesicDistance().compute_distance(manifold_projection.u[0], manifold_projection.u[1])
print(distance01)


# %% md
#
# Compute the pairwise distances for :math:`\Psi` and :math:`\Phi`, the left and right -singular eigenvectors,
# respectively, of singular value decomposition of each solution. In this case, use an user defined class
# `UserDistance`.

# %%

class UserDistance(GrassmannianDistance):

    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint):
        GrassmannianDistance.check_rows(xi, xj)

        rank_i = xi.data.shape[1]
        rank_j = xj.data.shape[1]

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        return np.sqrt(abs(rank_i - rank_j) * np.pi ** 2 / 4 + np.sum(theta ** 2))


pairwise_distance_psi = UserDistance().calculate_distance_matrix(manifold_projection.u, p_dim=p_dim)

pairwise_distance_phi = UserDistance().calculate_distance_matrix(manifold_projection.v, p_dim=p_dim)
print(pairwise_distance_psi)
print(pairwise_distance_phi)
