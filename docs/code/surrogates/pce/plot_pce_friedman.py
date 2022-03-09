"""

Friedman function (5 random inputs, scalar output)
======================================================================

In this example, PCE is used to generate a surrogate model for a given set of 5D data.

Friedman function
----------------------------------------

.. math:: f(x) = 10 sin(\pi x_1x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5

**Description:**  Dimensions: 5

**Input Domain:**  This function is evaluated on the hypercube :math:`x_i \in [0,1]` for all i = 1, …, 5.

**Reference:**  Friedman, J. H. (1991). Multivariate adaptive regression splines. The annals of statistics, 19(1), 1-67.
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
    return 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.5)**2 + 10*x[:,3] + 5*x[:,4]

# %% md
#
# Create a distribution object, generate samples and evaluate the function at the samples.

# %%

np.random.seed(1)

dist = Uniform(loc=0, scale=1)

marg = [dist]*5
joint = JointIndependent(marginals=marg)

n_samples = 200
x = joint.rvs(n_samples)
y = function(x)

# %% md
#
# Create an object from the PCE class. Then compute PCE coefficients using least squares regression.

# %%

max_degree = 3

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

n_samples = 100
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