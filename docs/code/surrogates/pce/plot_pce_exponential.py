"""

Exponential function (3 random inputs, scalar output)
======================================================================

In this example, PCE is used to generate a surrogate model for a given set of 3D data.

Dette & Pepelyshev exponential function
----------------------------------------

.. math:: f(x) = 100(\exp{(-2/x_1^{1.75})} + \exp{(-2/x_2^{1.5})} + \exp{(-2/x_3^{1.25})})

**Description:**  Dimensions: 3

**Input Domain:**  This function is evaluated on the hypercube :math:`x_i \in [0,1]` for all :math:`i = 1,2,3`.

**Reference:**  Dette, H., & Pepelyshev, A. (2010). Generalized Latin hypercube design for computer experiments. Technometrics, 52(4).
"""

# %% md
#
# Import necessary libraries.

# %%

import math
import numpy as np
import matplotlib.pyplot as plt
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *

# %% md
#
# Define the function.

# %%

def function(x):
    return 100*(np.exp(-2/(x[:,0]**1.75)) + np.exp(-2/(x[:,1]**1.5)) + np.exp(-2/(x[:,2]**1.25)))

# %% md
#
# Define the input probability distributions.

# %%

# input distributions
dist = Uniform(loc=0, scale=1)
marg = [dist]*3
joint = JointIndependent(marginals=marg)

# %% md
#
# Compute reference mean and variance values using Monte Carlo sampling.

# %%

# reference moments via Monte Carlo Sampling
n_samples_mc = 1000000
xx = joint.rvs(n_samples_mc)
yy = function(xx)
mean_ref = yy.mean()
var_ref = yy.var()

# %% md
#
# Create validation data sets, to be used later to estimate the accuracy of the PCE.

# %%


# validation data sets
n_samples_val = 1000
xx_val = joint.rvs(n_samples_val)
yy_val = function(xx_val)

# %% md
#
# Assess the PCE in terms of approximation and moment estimation accuracy, for increasing maximum polynomial degree.

# %%

# construct PCE surrogate models
l2_err = []
mean_err = []
var_err = []
for max_degree in range(1, 10):
    print(' ')

    # PCE basis
    print('Total degree: ', max_degree)
    polynomial_basis = TotalDegreeBasis(joint, max_degree)
    print('Size of basis:', polynomial_basis.polynomials_number)

    # generate training data
    sampling_coeff = 5
    print('Sampling coefficient: ', sampling_coeff)
    np.random.seed(42)
    n_samples = math.ceil(sampling_coeff * polynomial_basis.polynomials_number)
    print('Training data: ', n_samples)
    xx_train = joint.rvs(n_samples)
    yy_train = function(xx_train)

    # fit model
    least_squares = LeastSquareRegression()
    pce_metamodel = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce_metamodel.fit(xx_train, yy_train)

    # validation errors
    yy_val_pce = pce_metamodel.predict(xx_val)
    errors = np.abs(yy_val - yy_val_pce.flatten())
    error_l2 = np.linalg.norm(errors)
    l2_err.append(error_l2)
    print('Validation error in the l2 norm:', error_l2)

    # moment errors
    pce_moments = pce_metamodel.get_moments()
    mean_pce = pce_moments[0]
    var_pce = pce_moments[1]
    mean_err.append(np.abs((mean_pce - mean_ref) / mean_ref))
    var_err.append(np.abs((var_pce - var_ref) / var_ref))
    print('Relative error in the mean:', mean_err[-1])
    print('Relative error in the variance:', var_err[-1])
