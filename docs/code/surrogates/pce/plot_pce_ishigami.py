"""

Ishigami function (3 random inputs, scalar output)
======================================================================

In this example, we approximate the well-known Ishigami function with a total-degree Polynomial Chaos Expansion.

"""

# %% md
#
# Import necessary libraries.

# %%

import numpy as np
import math
import numpy as np
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *

# %% md
#
# We then define the Ishigami function, which reads:
# ..math:: f(x_1, x_2, x_3) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)

# %%

# function to be approximated
def ishigami(xx):
    """Ishigami function"""
    a = 7
    b = 0.1
    term1 = np.sin(xx[0])
    term2 = a * np.sin(xx[1])**2
    term3 = b * xx[2]**4 * np.sin(xx[0])
    return term1 + term2 + term3

# %% md
#
# The Ishigami function has three random inputs, which are uniformly distributed in :math:`[-\pi, \pi]`. Moreover, the
# input random variables are mutually independent, which simplifies the construction of the joint distribution. Let's
# define the corresponding distributions.

# %%

# input distributions
dist1 = Uniform(loc=-np.pi, scale=2*np.pi)
dist2 = Uniform(loc=-np.pi, scale=2*np.pi)
dist3 = Uniform(loc=-np.pi, scale=2*np.pi)
marg = [dist1, dist2, dist3]
joint = JointIndependent(marginals=marg)

# %% md
#
# We now define our PCE. Only thing we need is the joint distribution.
#
# We must now select a polynomial basis. Here we opt for a total-degree (TD) basis, such that the univariate
# polynomials have a maximum degree equal to $P$ and all multivariate polynomial have a total-degree
# (sum of degrees of corresponding univariate polynomials) at most equal to $P$. The size of the basis is then
# given by :math:`\frac{(N+P)!}{N! P!}`
# where :math:`N` is the number of random inputs (here, :math:`N+3`).

# %%

# maximum polynomial degree
P = 6

# construct total-degree polynomial basis
polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, P)

# check the size of the basis
print('Size of PCE basis:', polynomial_basis.polynomials_number)

# %% md
#
# We must now compute the PCE coefficients. For that we first need a training sample of input random variable
# realizations and the corresponding model outputs. These two data sets form what is also known as an
# ''experimental design''. It is generally advisable that the experimental design has $2-10$ times more data points
# than the number of PCE polynomials.

# %%

# create training data
sample_size = int(polynomial_basis.polynomials_number*5)
print('Size of experimental design:', sample_size)

# realizations of random inputs
xx_train = joint.rvs(sample_size)
# corresponding model outputs
yy_train = np.array([ishigami(x) for x in xx_train])


# %% md
#
# We now fit the PCE coefficients by solving a regression problem. There are multiple ways to do this, e.g. least
# squares regression, ridge regression, LASSO regression, etc. Here we opt for the _np.linalg.lstsq_ method, which
# is based on the _dgelsd_ solver of LAPACK.

# %%

# fit model
least_squares = LeastSquareRegression()
pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

pce.fit(xx_train, yy_train)

# %% md
#
# By simply post-processing the PCE's terms, we are able to get estimates regarding the mean and standard deviation
# of the model output.

# %%

mean_est = pce.get_moments()[0]
var_est = pce.get_moments()[1]
print('PCE mean estimate:', mean_est)
print('PCE variance estimate:', var_est)


# %% md
#
# Similarly to the mean and variance estimates, we can very simply estimate the Sobol sensitivity indices, which
# quantify the importance of the input random variables in terms of impact on the model output.

# %%

from UQpy.sensitivity import *
pce_sensitivity = PceSensitivity(pce)
sobol_first = pce_sensitivity.first_order_indices()
sobol_total = pce_sensitivity.total_order_indices()
print('First-order Sobol indices:')
print(sobol_first)
print('Total-order Sobol indices:')
print(sobol_total)

# %% md
#
# The PCE should become increasingly more accurate as the maximum polynomial degree $P$ increases. We will test that
# by computing the mean absolute error (MAE) between the PCE's predictions and the true model outputs, given a
# validation sample of $10^5$ data points.

# %%

# validation data sets
np.random.seed(999)  # fix random seed for reproducibility
n_samples_val = 100000
xx_val = joint.rvs(n_samples_val)
yy_val = np.array([ishigami(x) for x in xx_val])

mae = []  # to hold MAE for increasing polynomial degree
for degree in range(16):
    # define PCE
    polynomial_basis = PolynomialBasis.create_total_degree_basis(joint, degree)
    least_squares = LeastSquareRegression()
    pce_metamodel = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

    # create training data
    np.random.seed(1)  # fix random seed for reproducibility
    sample_size = int(pce_metamodel.polynomials_number * 5)
    xx_train = joint.rvs(sample_size)
    yy_train = np.array([ishigami(x) for x in xx_train])

    # fit PCE coefficients
    pce_metamodel.fit(xx_train, yy_train)

    # compute mean absolute validation error
    yy_val_pce = pce_metamodel.predict(xx_val).flatten()
    errors = np.abs(yy_val.flatten() - yy_val_pce)
    mae.append(np.linalg.norm(errors, 1) / n_samples_val)

    print('Polynomial degree:', degree)
    print('Mean absolute error:', mae[-1])
    print(' ')


