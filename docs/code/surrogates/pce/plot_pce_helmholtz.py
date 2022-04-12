"""

2D Helmholtz eigenvalues (2 random inputs, vector-valued output)
======================================================================

In this example, PCE is used to generate a surrogate model for a given set of 2D data for a numerical model with
multi-dimensional outputs.
"""

# %% md
#
# Import necessary libraries.

# %%

import numpy as np
import matplotlib.pyplot as plt
import math
from UQpy.distributions import Normal, JointIndependent
from UQpy.surrogates import *

# %% md
#
# The analytical function below describes the eigenvalues of the 2D Helmholtz equation on a square.

# %%

def analytical_eigenvalues_2d(Ne, lx, ly):
    """
    Computes the first Ne eigenvalues of a rectangular waveguide with
    dimensions lx, ly

    Parameters
    ----------
    Ne : integer
         number of eigenvalues.
    lx : float
         length in x direction.
    ly : float
         length in y direction.

    Returns
    -------
    ev : numpy 1d array
         the Ne eigenvalues
    """
    ev = [(m * np.pi / lx) ** 2 + (n * np.pi / ly) ** 2 for m in range(1, Ne + 1)
          for n in range(1, Ne + 1)]
    ev = np.array(ev)

    return ev[:Ne]


# %% md
#
# Create a distribution object.

# %%

pdf_lx = Normal(loc=2, scale=0.02)
pdf_ly = Normal(loc=1, scale=0.01)
margs = [pdf_lx, pdf_ly]
joint = JointIndependent(marginals=margs)

# %% md
#
# Define the number of input dimensions and choose the number of output dimensions (number of eigenvalues).

# %%

dim_in = 2
dim_out = 10

# %% md
#
# Construct PCE models by varying the maximum degree of polynomials (and therefore the number of polynomial basis) and
# compute the validation error for all resulting models.

# %%

errors = []
# construct PCE surrogate models
for max_degree in range(1, 6):
    print('Total degree: ', max_degree)
    polynomial_basis = TotalDegreeBasis(joint, max_degree)

    print('Size of basis:', polynomial_basis.polynomials_number)
    # training data
    sampling_coeff = 5
    print('Sampling coefficient: ', sampling_coeff)
    np.random.seed(42)
    n_samples = math.ceil(sampling_coeff * polynomial_basis.polynomials_number)
    print('Training data: ', n_samples)
    xx = joint.rvs(n_samples)
    yy = np.array([analytical_eigenvalues_2d(dim_out, x[0], x[1]) for x in xx])

    # fit model
    least_squares = LeastSquareRegression()
    pce_metamodel = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce_metamodel.fit(xx, yy)

    # coefficients
    # print('PCE coefficients: ', pce.C)

    # validation errors
    np.random.seed(999)
    n_samples = 1000
    x_val = joint.rvs(n_samples)
    y_val = np.array([analytical_eigenvalues_2d(dim_out, x[0], x[1]) for x in x_val])
    y_val_pce = pce_metamodel.predict(x_val)
    errors.append(np.linalg.norm((y_val - y_val_pce) / y_val, ord=1, axis=0))
    print('Relative absolute errors: ', errors[-1])
    print('')

# %% md
#
# Plot errors.

# %%

errors = np.array(errors)
plt.figure(1)
for i in range(np.shape(errors)[0]):
    plt.semilogy(np.linspace(1, dim_out, dim_out), errors[i], '--o', label='pol. degree: {}'.format(i+1))
plt.legend()
plt.show()

# %% md
#
# Moment estimation (directly estimated from the last PCE metamodel).

# %%

print('Mean PCE estimate:', pce_metamodel.get_moments()[0])
print('')
print('Variance PCE estimate:', pce_metamodel.get_moments()[1])