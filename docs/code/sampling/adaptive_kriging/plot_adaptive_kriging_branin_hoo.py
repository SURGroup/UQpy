"""

Expected Improvement - Branin Hoo
============================================

In this example, Monte Carlo Sampling is used to generate samples from Uniform distribution and new samples are
generated adaptively, using EIF (Expected Improvement Function) as the learning criteria.
"""

# %% md
#
# Branin-Hoo function
# --------------------
# Decription:
#
# >  - Dimensions: 2
# >  - This function is usually evaluated on the square $x_1 \in [-5, 10], \ x_2 \in [0, 15]$
# >  - The function has two local minima and one global minimum
# >  - Reference: Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling: a practical guide. Wiley.
#
# > $\displaystyle f(x) = a(x_2-bx_1^2 + cx_1 -r)^2 + s(1-t)\cos(x_1) + s + 5x_1$
# > <br>
# > <br>
# > where the recommended values of a, b, c, r, s and t are: $a = 1,\ b = 5.1/(4\pi^2),\ c = 5/\pi, \ r = 6, \ s = 10, \ t = 1/(8\pi)$
#
# <img src="branin.png" alt="branin.png" height="350" width="400" align=left>

# %%

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy, matplotlib and other necessary
# library for plots, but also need to import the MCS, AKMCS, Kriging and RunModel class from UQpy.

# %%
import shutil

import numpy as np
from matplotlib import pyplot as plt

from UQpy.surrogates import Kriging
from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.RunModel import RunModel
from UQpy.distributions import Uniform
from local_BraninHoo import function
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

# %% md
#
# Using UQpy MCS class to generate samples for two random variables, which are uniformly distributed

# %%

marginals = [Uniform(loc=-5, scale=15), Uniform(loc=0, scale=15)]
x = MonteCarloSampling(distributions=marginals, nsamples=20)

# %% md
#
# RunModel class is used to define an object to evaluate the model at sample points.

# %%

rmodel = RunModel(model_script='local_BraninHoo.py', vec=False)

# %% md
#
# Krig class defines an object to generate an surrogate model for a given set of data.

# %%

from UQpy.surrogates.kriging.regression_models import Linear
from UQpy.surrogates.kriging.correlation_models import Exponential
optimizer = MinimizeOptimizer(method="L-BFGS-B")
K = Kriging(regression_model=Linear(), correlation_model=Exponential(),optimizer=optimizer,
            correlation_model_parameters=[1, 1], optimizations_number=10)

# %% md
#
# Choose an appropriate learning function.

# %%

from UQpy.sampling.adaptive_kriging_functions.ExpectedImprovement import ExpectedImprovement

# %% md
#
# AKMCS class is used to generate new sample using 'U-function' as active learning function.

# %%

start_time = time.time()
learning_function= ExpectedImprovement()
a = AdaptiveKriging(runmodel_object=rmodel, samples=x.samples, surrogate=K,
                    learning_nsamples=10 ** 3, n_add=1,
                    learning_function=learning_function, distributions=marginals)
a.run(nsamples=50)
elapsed_time = time.time() - start_time

# %% md
#
# Visualize initial and new samples on top of the Branin-Hoo surface.

# %%

num = 200
xlist = np.linspace(-6, 11, num)
ylist = np.linspace(-1, 16, num)
X, Y = np.meshgrid(xlist, ylist)

Z = np.zeros((num, num))
for i in range(num):
    for j in range(num):
        tem = np.array([[X[i, j], Y[i, j]]])
        Z[i, j] = function(tem)

shutil.rmtree(rmodel.model_dir)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, Z, 10)
plt.xlabel('x1')
plt.ylabel('x2')
fig.colorbar(cp)
nd = x.nsamples
plt.scatter(a.samples[nd:, 0], a.samples[nd:, 1], color='pink', label='New samples')
plt.scatter(x.samples[:nd, 0], x.samples[:nd, 1], color='Red', label='Initial samples')
plt.title('Branin-Hoo function');
plt.legend()




