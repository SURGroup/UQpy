"""

Kernel
==================================

This example shows how to use the UQpy Grassmann class to compute kernels
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
from UQpy.dimension_reduction.grassman.manifold_projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction import Grassmann
import sys
from UQpy import ProjectionKernel
from UQpy import KernelComposition
from UQpy import OrthoMatrixPoints
from UQpy import Kernel

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
# Instatiate the UQpy class Grassmann considering the `projection_kernel` as the a kernel definition on the Grassmann
# manifold.

#%%

manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize,
                                    kernel_composition=KernelComposition.LEFT)
manifold = Grassmann(manifold_projected_points=manifold_projection)

#%% md
#
# Compute the kernels for $\Psi$ and $\Phi$, the left and right -singular eigenvectors, respectively, of singular value
# decomposition of each solution.

#%%

kernel_psi = manifold.evaluate_kernel_matrix(kernel=ProjectionKernel())

manifold_projection.kernel_composition = KernelComposition.RIGHT
kernel_phi = manifold.evaluate_kernel_matrix(kernel=ProjectionKernel())

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.title.set_text('kernel_psi')
ax1.imshow(kernel_psi)
ax2.title.set_text('kernel_phi')
ax2.imshow(kernel_phi)
plt.show()

#%% md
#
# Compute the kernel only for 2 points.

#%%

grassmann_points = OrthoMatrixPoints(input_points=[manifold_projection.psi[0],
                                                   manifold_projection.psi[1],
                                                   manifold_projection.psi[2]],
                                     p_planes_dimensions=manifold_projection.p_planes_dimensions)
manifold = Grassmann(manifold_projected_points=grassmann_points)
kernel_01 = manifold.evaluate_kernel_matrix(kernel=ProjectionKernel())

fig = plt.figure()
plt.imshow(kernel_01)
plt.show()

#%% md
#
# Compute the kernels for $\Psi$ and $\Phi$, the left and right -singular eigenvectors, respectively, of singular value
# decomposition of each solution. In this case, use an user defined function `my_kernel`.

#%%

class UserKernel(Kernel):

    def apply_method(self, data):
        data.evaluate_matrix(self, self.kernel_operator)

    def pointwise_operator(self, x0, x1):
        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        r = np.dot(x0.T, x1)
        det = np.linalg.det(r)
        ker = det * det
        return ker

manifold_projection.kernel_composition = KernelComposition.LEFT
kernel_user_psi = manifold.evaluate_kernel_matrix(kernel=UserKernel())

manifold_projection.kernel_composition = KernelComposition.RIGHT
kernel_user_phi = manifold.evaluate_kernel_matrix(kernel=UserKernel())


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.title.set_text('kernel_psi')
ax1.imshow(kernel_user_psi)
ax2.title.set_text('kernel_phi')
ax2.imshow(kernel_user_phi)
plt.show()
