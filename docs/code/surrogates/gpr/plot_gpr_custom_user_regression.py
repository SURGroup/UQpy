"""

User defined Regression & Correlation
======================================================================

In this example, Kriging is used to generate a surrogate model for a given data. In this data, sample points are
generated using STS class and functional value at sample points are estimated using a model defined in python script
s('python_model_function.py).
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the STS, RunModel and Krig class from UQpy.

# %%
import shutil

from UQpy.surrogates import Kriging
from UQpy.sampling import RectangularStrata
from UQpy.sampling import TrueStratifiedSampling
from UQpy.run_model.RunModel import RunModel
from UQpy.surrogates.kriging.regression_models.baseclass import Regression
from UQpy.surrogates.kriging.correlation_models.baseclass import Correlation
from UQpy.distributions import Gamma
import numpy as np
import matplotlib.pyplot as plt

# %% md
#
# Create a distribution object.

# %%

marginals = [Gamma(a=2., loc=1., scale=3.)]

# %% md
#
# Create a strata object.

# %%

strata = RectangularStrata(strata_number=[20])

# %% md
#
# Using UQpy STS class to generate samples for two random variables, which are uniformly distributed between 0 and 1.

# %%

x = TrueStratifiedSampling(distributions=marginals, strata_object=strata,
                           nsamples_per_stratum=1, random_state=1)

# %% md
#
# RunModel is used to evaluate function values at sample points. Model is defined as a function in python file
# 'python_model_function.py'.

# %%

rmodel = RunModel(model_script='local_python_model_1Dfunction.py', delete_files=True)
rmodel.run(samples=x.samples)


# %% md
#
# A regression model is defined, this function return the basis function and its jacobian.

# %%

class UserRegression(Regression):
    def r(self, s):
        fx = np.concatenate((np.ones([np.size(s, 0), 1]), s), 1)
        jf_b = np.zeros([np.size(s, 0), np.size(s, 1), np.size(s, 1)])
        np.einsum('jii->ji', jf_b)[:] = 1
        jf = np.concatenate((np.zeros([np.size(s, 0), np.size(s, 1), 1]), jf_b), 2)
        return fx, jf


# %% md
#
# A user-defined correlation model is created, which returns covariance matrix and its derivatives.

# %%

class UserCorrelation(Correlation):
    def c(self, x, s, params, dt=False, dx=False):
        x, s = np.atleast_2d(x), np.atleast_2d(s)
        # Create stack matrix, where each block is x_i with all s
        stack = - np.tile(np.swapaxes(np.atleast_3d(x), 1, 2), (1, np.size(s, 0), 1)) + np.tile(s,
                                                                                                (np.size(x, 0), 1,
                                                                                                 1))
        rx = np.exp(np.sum(-params * (stack ** 2), axis=2))
        if dt:
            drdt = -(stack ** 2) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
            return rx, drdt
        if dx:
            drdx = 2 * params * stack * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
            return rx, drdx
        return rx


# %% md
#
# Using UQpy Krig class to generate a surrogate for generated data. In this illustration, user defined regression model
# and correlation model are used.

# %%

regression_model = UserRegression()
correlation_model = UserCorrelation()

from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

optimizer = MinimizeOptimizer(method="L-BFGS-B")

K = Kriging(regression_model=regression_model, optimizer=optimizer,
            correlation_model=correlation_model,
            optimizations_number=10,
            correlation_model_parameters=[10],
            random_state=2)
K.fit(samples=x.samples, values=rmodel.qoi_list)
print(K.correlation_model_parameters)


# %% md
#
# Kriging surrogate is used to compute the response surface and its gradient.

# %%

num = 1000
x1 = np.linspace(min(x.samples), max(x.samples), num)
y, mse = K.predict(x1.reshape([num, 1]), return_std=True)
y_grad = K.jacobian(x1.reshape([num, 1]))

# %% md
#
# Actual model is evaluated at all points to compare it with kriging surrogate.

# %%

rmodel.run(samples=x1, append_samples=False)
shutil.rmtree(rmodel.model_dir)

# %% md
#
# This plot shows the input data as blue dot, blue curve is actual function and orange curve represents response curve.
# This plot also shows the gradient and 95% confidence interval of the kriging surrogate.

# %%

fig = plt.figure()
ax = plt.subplot(111)
plt.plot(x1, rmodel.qoi_list, label='Sine')
plt.plot(x1, y, label='Surrogate')
plt.plot(x1, y_grad, label='Gradient')
plt.scatter(K.samples, K.values, label='Data')
plt.fill(np.concatenate([x1, x1[::-1]]),
         np.concatenate([y - 1.9600 * mse,
                         (y + 1.9600 * mse)[::-1]]),
         alpha=.5, fc='y', ec='None', label='95% CI')
# plt.legend(loc='lower right')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
