r"""

Ishigami function
==============================================

The ishigami function is a non-linear, non-monotonic function that is commonly used to
benchmark uncertainty and senstivity analysis methods.

.. math::
    f(x_1, x_2, x_3) = sin(x_1) + a \cdot sin^2(x_2) + b \cdot x_3^4 sin(x_1)

.. math::
    x_1, x_2, x_3 \sim \mathcal{U}(-\pi, \pi), \quad a, b\in \mathbb{R}

First order Sobol indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    S_1 = \frac{V_1}{\mathbb{V}[Y]}, \quad S_2 = \frac{V_2}{\mathbb{V}[Y]}, \quad S_3 = \frac{V_3}{\mathbb{V}[Y]} = 0,

.. math::
    V_1 = 0.5 (1 + \frac{b\pi^4}{5})^2, \quad V_2 = \frac{a^2}{8}, \quad V_3 = 0

.. math::
    \mathbb{V}[Y] = \frac{a^2}{8} + \frac{b\pi^4}{5} + \frac{b^2\pi^8}{18} + \frac{1}{2}


"""

# %%

import numpy as np
from UQpy.distributions import Uniform
from UQpy.sampling import MonteCarloSampling
from UQpy.sensitivity.GPSobolSensitivity import GPSobolSensitivity
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor

# %% md
#
# Consider the following function :math:`f(x)`.
#
# .. math:: f(x) = \sin{x_1} + 7 \sin^2{x_2} + 0.1 x_3^4 \sin{x_1}, \quad \quad x \in [0,1]

# %%


def ishigami(x):
    return np.sin(x[:, 0]) + 7*np.sin(x[:, 1])**2 + 0.1*(x[:, 2]**4)*np.sin(x[:, 0])


# %% md
#
# Define marginal distribution of input variables and generate sample data to compute sobol sensitivity

dimension = 3
dist_object = [Uniform(-np.pi, 2 * np.pi)] * dimension
samples = MonteCarloSampling(distributions=dist_object, nsamples=200, random_state=12).samples
values = ishigami(samples)


# %% md
#
# Train Gaussian Process Regressor on the data
kernel = ConstantKernel(constant_value_bounds=(1e-2, 5)) * RBF(length_scale=[0.5] * dimension,
                                                               length_scale_bounds=(0.1, 20))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(samples, values)


# %% md
#
# Compute Sobol indices

sobol = GPSobolSensitivity(surrogate=gpr, distributions=dist_object)
sobol.run(samples=samples, values=values)

# %% [markdown]
# **First order Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1` = 0.3139
#
# :math:`S_2` = 0.4424
#
# :math:`S_3` = 0.0

# %%
# GP based estimates for first order Sobol indices:
print(sobol.sobol_mean)
