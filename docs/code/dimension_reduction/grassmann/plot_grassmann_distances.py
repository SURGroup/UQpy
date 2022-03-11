"""

Compute distances between points
==================================

This example shows how to use the UQpy Grassmann class to compute distances
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the Grassmann class from UQpy implemented in the DimensionReduction module.

#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UQpy import SvdProjection
import sys
from UQpy import GrassmannDistance, RiemannianDistance

from UQpy.dimension_reduction import Grassmann

#%% md
#
# Generate four random matrices with reduced rank corresponding to the different samples. The samples are stored in
# `matrices`.

#%%

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

#%% md
#
# Instatiate the UQpy class Grassmann considering the `"grassmann_distance"` as the a definition of distance on the
# manifold.

#%%

manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

#%% md
#
# Compute the pairwise distances for $\Psi$ and $\Phi$, the left and right -singular eigenvectors, respectively, of
# singular value decomposition of each solution.

#%%

pairwise_distance = Grassmann.calculate_pairwise_distances(distance_method=GrassmannDistance(),
                                                           points_grassmann=manifold_projection.psi)
print(pairwise_distance)

#%% md
#
# Compute the distance between 2 points.

#%%

distance_metric = GrassmannDistance()
distance01 = distance_metric.compute_distance(manifold_projection.psi[0], manifold_projection.psi[1])
print(distance01)

#%% md
#
# Compute the pairwise distances for $\Psi$ and $\Phi$, the left and right -singular eigenvectors, respectively, of
# singular value decomposition of each solution. In this case, use an user defined function `my_distance`.

#%%

class UserDistance(RiemannianDistance):

    def compute_distance(self, x0, x1):
        """
            Estimate the user distance.

            **Input:**

            * **x0** (`list` or `ndarray`)
                Point on the grassman manifold.

            * **x1** (`list` or `ndarray`)
                Point on the grassman manifold.

            **Output/Returns:**

            * **distance** (`float`)
                Procrustes distance between x0 and x1.
            """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        # (ui, si, vi) = svd(r, rank)

        ui, si, vi = np.linalg.svd(r, full_matrices=True, hermitian=False)  # Compute the SVD of matrix
        si = np.diag(si)  # Transform the array si into a diagonal matrix containing the singular values
        vi = vi.T  # Transpose of vi

        u = ui[:, :rank]
        s = si[:rank, :rank]
        v = vi[:, :rank]

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = np.sin(theta / 2) ** 2
        distance = np.sqrt(abs(k - l) + 2 * np.sum(theta))

        return distance

pairwise_distance_psi = \
    Grassmann.calculate_pairwise_distances(distance_method=UserDistance(),
                                           points_grassmann=manifold_projection.psi)
pairwise_distance_phi = \
    Grassmann.calculate_pairwise_distances(distance_method=UserDistance(),
                                           points_grassmann=manifold_projection.phi)
print(pairwise_distance_psi)
print(pairwise_distance_phi)