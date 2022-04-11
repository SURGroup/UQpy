"""

Sinusoidal Function  Sphere function (2 random inputs, scalar output)
======================================================================

In this example, PCE is used to generate a surrogate model for a given set of 2D data.

.. math:: f(x) = x_1^2 + x_2^2

**Description:**  Dimensions: 2

**Input Domain:**  This function is evaluated on the hypercube :math:`x_i \in [-5.12, 5.12]` for all :math:`i = 1,2`.

**Global minimum:** :math:`f(x^*)=0,` at :math:`x^* = (0,0)`.

**Reference:**  Dixon, L. C. W., & Szego, G. P. (1978). The global optimization problem: an introduction. Towards global optimization, 2, 1-15.
"""

# %% md
#
# Import necessary libraries.

# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from UQpy.surrogates import *
from UQpy.distributions import Uniform, JointIndependent

# %% md
#
# Define the function.

# %%

def function(x,y):
    return x**2 + y**2

# %% md
#
# Create a distribution object, generate samples and evaluate the function at the samples.

# %%

np.random.seed(1)

dist_1 = Uniform(loc=-5.12, scale=10.24)
dist_2 = Uniform(loc=-5.12, scale=10.24)

marg = [dist_1, dist_2]
joint = JointIndependent(marginals=marg)

n_samples = 100
x = joint.rvs(n_samples)
y = function(x[:,0], x[:,1])

# %% md
#
# Visualize the 2D function.

# %%

xmin, xmax = -6,6
ymin, ymax = -6,6
X1 = np.linspace(xmin, xmax, 50)
X2 = np.linspace(ymin, ymax, 50)
X1_, X2_ = np.meshgrid(X1, X2) # grid of points
f = function(X1_, X2_)

fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1_, X2_, f, rstride=1, cstride=1, cmap='gnuplot2', linewidth=0, antialiased=False)
ax.set_title('True function')
ax.set_xlabel('$x_1$', fontsize=15)
ax.set_ylabel('$x_2$', fontsize=15)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(20, 140)
fig.colorbar(surf, shrink=0.5, aspect=7)

plt.show()

# %% md
#
# Visualize training data.

# %%

fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')
ax.scatter(x[:,0], x[:,1], y, s=20, c='r')

ax.set_title('Training data')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(20,140)
ax.set_xlabel('$x_1$', fontsize=15)
ax.set_ylabel('$x_2$', fontsize=15)
plt.show()

# %% md
#
# Create an object from the PCE class. Compute PCE coefficients using least squares regression.

# %%

max_degree = 3
polynomial_basis = TotalDegreeBasis(joint, max_degree)
least_squares = LeastSquareRegression()
pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

pce.fit(x,y)

# %% md
#
# Compute PCE coefficients using LASSO.

# %%

polynomial_basis = TotalDegreeBasis(joint, max_degree)
lasso = LassoRegression()
pce2 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=lasso)

pce2.fit(x,y)

# %% md
#
# Compute PCE coefficients with Ridge regression.

# %%

polynomial_basis = TotalDegreeBasis(joint, max_degree)
ridge = RidgeRegression()
pce3 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=ridge)

pce3.fit(x,y)

# %% md
#
# PCE surrogate is used to predict the behavior of the function at new samples.

# %%

n_test_samples = 10000
x_test = joint.rvs(n_test_samples)
y_test = pce.predict(x_test)


# %% md
#
# Plot PCE prediction.

# %%

fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], y_test, s=1)

ax.set_title('PCE predictor')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(20,140)
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
ax.set_xlabel('$x_1$', fontsize=15)
ax.set_ylabel('$x_2$', fontsize=15)
plt.show()

# %% md
# Error Estimation
# -----------------
# Construct a validation dataset and get the validation error.

# %%

# validation sample
n_samples = 150
x_val = joint.rvs(n_samples)
y_val = function(x_val[:,0], x_val[:,1])

# PCE predictions
y_pce  = pce.predict(x_val).flatten()
y_pce2 = pce2.predict(x_val).flatten()
y_pce3 = pce3.predict(x_val).flatten()

# mean relative validation errors
error = np.sum(np.abs((y_val - y_pce)/y_val))/n_samples
error2 = np.sum(np.abs((y_val - y_pce2)/y_val))/n_samples
error3 = np.sum(np.abs((y_val - y_pce3)/y_val))/n_samples

print('Mean rel. error, LSTSQ:', error)
print('Mean rel. error, LASSO:', error2)
print('Mean rel. error, Ridge:', error3)

# %% md
# Moment Estimation
# -----------------
# Returns mean and variance of the PCE surrogate.

# %%

n_mc = 1000000
x_mc = joint.rvs(n_mc)
y_mc = function(x_mc[:,0], x_mc[:,1])
mean_mc = np.mean(y_mc)
var_mc = np.var(y_mc)

print('Moments from least squares regression :', pce.get_moments())
print('Moments from LASSO regression :', pce2.get_moments())
print('Moments from Ridge regression :', pce3.get_moments())
print('Moments from Monte Carlo integration: ', mean_mc, var_mc)