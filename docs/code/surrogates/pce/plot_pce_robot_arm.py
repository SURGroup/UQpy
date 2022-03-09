"""

Robot Arm function (8 random inputs, scalar output)
======================================================================

In this example, PCE is used to generate a surrogate model for a given set of 8D data.

<img src="Example_RobotArm_function.png" alt="Drawing" style="width: 200px;"/>

**Dimensions:** 8

**Description:**  Models the position of a robot arm which has four segments.

**Input Domain:**  The input variables and their usual input ranges are: :math:`[0,1]` for the :math:`L_i` inputs
and :math:`[0,2\pi]` for the :math:`\theta_i` inputs.

**Function:**

.. math:: f(\textbf{x}) = \Big( \sum_{i=1}^{4}L_i cos\Big(\sum_{j=1}^{i} \theta_j\Big) \Big)^{2} + \Big( \sum_{i=1}^{4}L_i sin\Big(\sum_{j=1}^{i} \theta_j\Big) \Big)^{2}

**Output:** The square distance from the end of the robot arm to the origin.

**Reference:**  An, J., & Owen, A. (2001). Quasi-regression. Journal of Complexity, 17(4), 588-607.

"""

# %% md
#
# Import necessary libraries.

# %%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *

# %% md
#
# Define the function.

# %%

def function(x):
    # without square root
    u1 = x[:, 4] * np.cos(x[:, 0])
    u2 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1))
    u3 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1))
    u4 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.cos(np.sum(x[:, :4], axis=1))

    v1 = x[:, 4] * np.sin(x[:, 0])
    v2 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1))
    v3 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1))
    v4 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.sin(np.sum(x[:, :4], axis=1))

    return (u1 + u2 + u3 + u4) ** 2 + (v1 + v2 + v3 + v4) ** 2

# %% md
#
# Create a distribution object, generate samples and evaluate the function at the samples.

# %%

np.random.seed(1)

dist_1 = Uniform(loc=0, scale=2*np.pi)
dist_2 = Uniform(loc=0, scale=1)

marg = [dist_1]*4
marg_1 = [dist_2]*4
marg.extend(marg_1)

joint = JointIndependent(marginals=marg)

n_samples = 9000
x = joint.rvs(n_samples)
y = function(x)

# %% md
#
# Create an object from the PCE class. Compute PCE coefficients using least squares regression.

# %%

max_degree = 6
polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, max_degree)
least_squares = LeastSquareRegression()
pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

pce.fit(x,y)

# %% md
#
# Compute PCE coefficients using Lasso regression.

# %%

polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, max_degree)
lasso = LassoRegression()
pce2 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=lasso)

pce2.fit(x,y)

# %% md
#
# Compute PCE coefficients using Ridge regression.

# %%

polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, max_degree)
ridge = RidgeRegression()
pce3 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=ridge)

pce3.fit(x,y)

# %% md
# Error Estimation
# -----------------
# Validation error.

# %%

n_samples = 500
x_val = joint.rvs(n_samples)
y_val = function(x_val)

y_pce = pce.predict(x_val).flatten()
y_pce2 = pce2.predict(x_val).flatten()
y_pce3 = pce3.predict(x_val).flatten()

error = np.sum(np.abs(y_pce - y_val)/np.abs(y_val))/n_samples
error2 = np.sum(np.abs(y_pce2 - y_val)/np.abs(y_val))/n_samples
error3 = np.sum(np.abs(y_pce3 - y_val)/np.abs(y_val))/n_samples

print('Validation error, LSTSQ-PCE:', error)
print('Validation error, LASSO-PCE:', error2)
print('Validation error, Ridge-PCE:', error3)

# %% md
# Moment Estimation
# -----------------
# Returns mean and variance of the PCE surrogate.

# %%

n_mc = 1000000
x_mc = joint.rvs(n_mc)
y_mc = function(x_mc)
mean_mc = np.mean(y_mc)
var_mc = np.var(y_mc)

print('Moments from least squares regression :', pce.get_moments())
print('Moments from LASSO regression :', pce2.get_moments())
print('Moments from Ridge regression :', pce3.get_moments())
print('Moments from MC integration: ', mean_mc, var_mc)