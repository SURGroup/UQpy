"""

Interpolation
==================================

This example shows how to use the UQpy Grassmann class to perform interpolation on the Grassmann manifold
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
import sys
from UQpy.dimension_reduction import Grassmann
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent
from UQpy.dimension_reduction.distances.grassmanian.GrassmanDistance import GrassmannDistance
from UQpy import Interpolation
from UQpy import LinearInterpolation
from UQpy import InterpolationMethod

#%% md
#
# Generate the initial samples located at the vertices of a triangle. The coordinates of each vertix are stored in
# `nodes` and `point` contain the point to be interpolated on the tangent space.

#%%

nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # node_0, node_1, node_2.
point = np.array([0.1, 0.1])  # Point to interpolate.

plot_ = nodes[0:]
Xplot = plot_.T[0].tolist()
Xplot.append(plot_[0][0])
Yplot = plot_.T[1].tolist()
Yplot.append(plot_[0][1])
plt.plot(Xplot, Yplot)
plt.plot(nodes[0][0], nodes[0][1], 'ro')
plt.plot(nodes[1][0], nodes[1][1], 'ro')
plt.plot(nodes[2][0], nodes[2][1], 'ro')
plt.plot(nodes[3][0], nodes[3][1], 'ro')
plt.plot(point[0], point[1], 'bo')

dt = 0.015
plt.text(nodes[0][0] + dt, nodes[0][1] + dt, '0')
plt.text(nodes[1][0] + dt, nodes[1][1] + dt, '1')
plt.text(nodes[2][0] + dt, nodes[2][1] + dt, '2')
plt.text(nodes[3][0] + dt, nodes[3][1] + dt, '2')
plt.text(point[0] + dt, point[1] + dt, 'point')
plt.show()
plt.close()

#%% md
#
# Generate three random matrices with reduced rank corresponding to the different samples. The samples are stored in
# `Solutions`.

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
Solutions = [Sol0, Sol1, Sol2, Sol3]

# Plot the solutions
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.title.set_text('Solution 0')
ax1.imshow(Sol0)
ax2.title.set_text('Solution 1')
ax2.imshow(Sol1)
ax3.title.set_text('Solution 2')
ax3.imshow(Sol2)
ax4.title.set_text('Solution 3')
ax4.imshow(Sol3)
plt.show()

#%% md
#
# Firs, let's perform the interpolation step-by-step using an object of the ``UQpy.Kriging`` class to interpolate.
# Further, instatiate the ``UQpy`` class ``Grassmann`` considering the `grassmann_distance` for the distance,
# `gradient_descent` to estimate the Karcher mean.

#%%

from UQpy.surrogates import Kriging
from UQpy.surrogates.kriging.regression_models.Linear import Linear
from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

Krig = Kriging(regression_model=Linear(),
               correlation_model=Exponential(),
               correlation_model_parameters=[1.0, 1.0],
               optimizations_number=1)

manifold_projection = SvdProjection(Solutions, p_planes_dimensions=sys.maxsize)

#%% md
#
# Compute the Karcher mean for $\Psi$ and $\Phi$, the left and right -singular eigenvectors, respectively, of singular
# value decomposition of each solution.

#%%

optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
karcher_psi = Grassmann.karcher_mean(points_grassmann=manifold_projection.psi,
                                  p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                  optimization_method=optimization_method,
                                  distance=GrassmannDistance())
karcher_phi = Grassmann.karcher_mean(points_grassmann=manifold_projection.phi,
                                     p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                     optimization_method=optimization_method,
                                     distance=GrassmannDistance())

#%% md
#
# Rearrange the singular values $\Sigma$ of each solution as a diagonal matrix.

#%%

sigma_m = []
for i in range(len(manifold_projection.sigma)):
    sigma_m.append(np.diag(manifold_projection.sigma[i]))

#%% md
#
# Select the Karcher mean as a reference point for $\Psi$ and $\Phi$ and map those points on the manifold to the
# tangent space generated at the reference point.

#%%

gammaPsi = Grassmann.log_map(manifold_projection.psi, reference_point=karcher_psi)
gammaPhi = Grassmann.log_map(manifold_projection.phi, reference_point=karcher_phi)

#%% md
#
# Perform the standard linear interpolation of `point` on the tangent space for $\Psi$, $\Phi$, and $\Sigma$. The
# interpolated points are given by $\tilde{\Psi}$, $\tilde{\Phi}$, and $\tilde{\Sigma}$. Thus, the interpolated
# solution is given by $\tilde{\mathrm{X}}=\tilde{\Psi}\tilde{\Phi}\tilde{\Sigma}$.

#%%

interpolation = Interpolation(interpolation_method=Krig)

interpPsi = interpolation.interpolate_sample(coordinates=nodes, samples=gammaPsi, point=point)
interpPhi = interpolation.interpolate_sample(coordinates=nodes, samples=gammaPhi, point=point)
interpS = interpolation.interpolate_sample(coordinates=nodes, samples=sigma_m, point=point)

PsiTilde = Grassmann.exp_map([interpPsi], reference_point=karcher_psi)
PhiTilde = Grassmann.exp_map([interpPhi], reference_point=karcher_phi)

PsiTilde = np.array(PsiTilde[0])
PhiTilde = np.array(PhiTilde[0])
SolTilde = np.dot(np.dot(PsiTilde, interpS), PhiTilde.T)

#%% md
#
# Print the interpolated solution and compare to the given solutions associated to the different vertices of the
# triangle.

#%%

print(Sol0)
print(" ")
print(Sol1)
print(" ")
print(Sol2)
print("-------------------------")
print(SolTilde)

plot_ = nodes[0:]
Xplot = plot_.T[0].tolist()
Xplot.append(plot_[0][0])
Yplot = plot_.T[1].tolist()
Yplot.append(plot_[0][1])
plt.plot(Xplot, Yplot)
plt.plot(nodes[0][0], nodes[0][1], 'ro')
plt.plot(nodes[1][0], nodes[1][1], 'ro')
plt.plot(nodes[2][0], nodes[2][1], 'ro')
plt.plot(nodes[3][0], nodes[3][1], 'ro')
plt.plot(point[0], point[1], 'bo')

dt = 0.015
plt.text(nodes[0][0] + dt, nodes[0][1] + dt, '0')
plt.text(nodes[1][0] + dt, nodes[1][1] + dt, '1')
plt.text(nodes[2][0] + dt, nodes[2][1] + dt, '2')
plt.text(nodes[3][0] + dt, nodes[3][1] + dt, '3')
plt.text(point[0] + dt, point[1] + dt, 'p')
plt.show()
plt.close()

# Plot the solutions
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.title.set_text('Solution 0')
ax1.imshow(Sol0)
ax2.title.set_text('Solution 1')
ax2.imshow(Sol1)
ax3.title.set_text('Solution 2')
ax3.imshow(Sol2)
ax4.title.set_text('Solution 3')
ax4.imshow(Sol3)
ax5.title.set_text('Interpolated')
ax5.imshow(SolTilde)
plt.show()

#%% md
#
# All the operations above are implemented in the method `interpolate`, which is used below. in this case, the
# interpolation is performed on the entries of the input matrices.

#%%

X = manifold_projection.reconstruct_solution(interpolation=interpolation,
                                             coordinates=nodes,
                                             point=point,
                                             p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                             distance=GrassmannDistance(),
                                             element_wise=True,
                                             optimization_method=GradientDescent())

# Plot the solutions
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.title.set_text('Solution 0')
ax1.imshow(Sol0)
ax2.title.set_text('Solution 1')
ax2.imshow(Sol1)
ax3.title.set_text('Solution 2')
ax3.imshow(Sol2)
ax4.title.set_text('Solution 3')
ax4.imshow(Sol3)
ax5.title.set_text('Interpolated')
ax5.imshow(X)
plt.show()

#%% md
#
# Now, let's use an object of sklearn.gaussiann_process. (To run this example, you have to install the scikit learn
# toolbox in advance.)

#%%

from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor()

# Instantiate the method again with: interp_object=gp.
interpolation = Interpolation(interpolation_method=gp)

X = manifold_projection.reconstruct_solution(coordinates=nodes, point=[point], element_wise=True,
                                             interpolation=interpolation,
                                             optimization_method=GradientDescent(),
                                             distance=GrassmannDistance(),
                                             p_planes_dimensions=manifold_projection.p_planes_dimensions)

#%% md
#
# Plot the solution.

#%%

# Plot the solutions
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.title.set_text('Solution 0')
ax1.imshow(Sol0)
ax2.title.set_text('Solution 1')
ax2.imshow(Sol1)
ax3.title.set_text('Solution 2')
ax3.imshow(Sol2)
ax4.title.set_text('Solution 3')
ax4.imshow(Sol3)
ax5.title.set_text('Interpolated')
ax5.imshow(X)
plt.show()
