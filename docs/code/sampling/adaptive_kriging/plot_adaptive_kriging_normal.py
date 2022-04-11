"""

U-Function & User-defined learning function
============================================

In this example, Monte Carlo Sampling is used to generate samples from Normal distribution and new samples are generated
adaptively, using U-function as the learning criteria .
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy, matplotlib and other necessary
# library for plots, but also need to import the :class:`.MonteCarloSampling`,
# :class:`.AdaptiveKriging`, :class:`.Kriging` and :class:`.RunModel` class from UQpy.

# %%
import shutil

from UQpy import PythonModel
from UQpy.surrogates.gaussian_process import GaussianProcessRegression
from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import Normal
from local_series import series
import matplotlib.pyplot as plt
import time
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer


# %% md
#
# Using UQpy :class:`.MonteCarloSampling` class to generate samples for two random variables, which are normally
# distributed with mean :math:`0` and variance :math:`1`.

# %%

marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)

# %% md
#
# RunModel class is used to define an object to evaluate the model at sample points.

# %%

model = PythonModel(model_script='local_series.py', model_object_name='series')
rmodel = RunModel(model=model)


# %% md
#
# :class:`.Kriging` class defines an object to generate a surrogate model for a given set of data.

# %%

from UQpy.surrogates.gaussian_process.regression_models import LinearRegression
from UQpy.surrogates.gaussian_process.kernels import RBF
bounds = [[10**(-3), 10**3], [10**(-3), 10**2], [10**(-3), 10**2]]
optimizer = MinimizeOptimizer(method="L-BFGS-B", bounds=bounds)
K = GaussianProcessRegression(regression_model=LinearRegression(), kernel=RBF(), optimizer=optimizer,
                              hyperparameters=[1, 1, 0.1], optimizations_number=10, noise=False)

# %% md
#
# This example works for all three learning function based on reliability analysis.
#
# :class:`.AdaptiveKriging` class is used to generate new sample using :class:`.UFunction` as active learning function.

# %%

from UQpy.sampling.adaptive_kriging_functions import *
start_time = time.time()
learning_function = WeightedUFunction(weighted_u_stop=2)
a = AdaptiveKriging(runmodel_object=rmodel, surrogate=K, learning_nsamples=10 ** 3, n_add=1,
                    learning_function=learning_function, distributions=marginals, random_state=2)
a.run(nsamples=100, samples=x.samples)

elapsed_time = time.time() - start_time


time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
g = a.surrogate.predict(a.learning_set, False)
n_ = a.learning_set.shape[0] + len(a.qoi)
pf = (sum(g < 0) + sum(np.array(a.qoi) < 0)) / n_
print('Time: ', elapsed_time)
print('Function evaluation: ', a.samples.shape[0])
print('Probability of failure: ', pf)

# %% md
#
# This figure shows the location of new samples generated using active learning function.

# %%

num = 50
x1 = np.linspace(-7, 7, num)
x2 = np.linspace(-7, 7, num)
x1v, x2v = np.meshgrid(x1, x2)
y = np.zeros([num, num])
y_act = np.zeros([num, num])
mse = np.zeros([num, num])
for i in range(num):
    for j in range(num):
        xa = marginals[0].cdf(np.atleast_2d(x1v[i, j]))
        ya = marginals[1].cdf(np.atleast_2d(x2v[i, j]))
        y[i, j] = a.surrogate.predict(np.hstack([xa, ya]))
        y_act[i, j] = series(np.array([[x1v[i, j], x2v[i, j]]]))

fig, ax = plt.subplots()
kr_a = ax.contour(x1v, x2v, y_act, levels=[0], colors='Black')

# Plot for scattered data
nd = x.nsamples
ID1 = ax.scatter(a.samples[nd:, 0], a.samples[nd:, 1], color='Grey', label='New samples')
ID = ax.scatter(x.samples[:nd, 0], x.samples[:nd, 1], color='Red', label='Initial samples')
plt.legend(handles=[ID1, ID])
plt.show()

# %% md
#
# User-defined Learning function
# ------------------------------

# %%

class UserLearningFunction(LearningFunction):

    def __init__(self, u_stop: int = 2):
        self.u_stop = u_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
        # AKMS class use these inputs to compute the learning function

        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:n_add]

        indicator = False
        if min(u[:, 0]) >= self.u_stop:
            indicator = True

        return population[rows, :], u[rows, 0], indicator

# %% md
#
# Creating new instances of :class:`.Kriging` and :class:`.RunModel` class.

# %%
bounds = [[10**(-3), 10**3], [10**(-3), 10**2], [10**(-3), 10**2]]
optimizer = MinimizeOptimizer(method="L-BFGS-B", bounds=bounds)
K1 = GaussianProcessRegression(regression_model=LinearRegression(), kernel=RBF(), optimizer=optimizer,
                               hyperparameters=[1, 1, 0.1], optimizations_number=1)
model = PythonModel(model_script='local_series.py', model_object_name='series')
rmodel1 = RunModel(model=model)

# %% md
#
# Executing :class:`Adaptivekriging` with the user-defined learning function.

# %%

start_time = time.time()
ak = AdaptiveKriging(runmodel_object=rmodel1, samples=x.samples, surrogate=K1, learning_nsamples=10 ** 3,
                     n_add=1, learning_function=UserLearningFunction(), distributions=marginals, random_state=3)
ak.run(nsamples=100)


elapsed_time = time.time() - start_time

time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
g = ak.surrogate.predict(ak.learning_set, False)
n_ = ak.learning_set.shape[0] + len(ak.qoi)
pf = (sum(g < 0) + sum(np.array(ak.qoi) < 0)) / n_
print('Time: ', elapsed_time)
print('Function evaluation: ', ak.samples.shape[0])
print('Probability of failure: ', pf)

# %% md
#
# This figure shows the location of new samples generated using active learning function.

# %%

fig1, ax1 = plt.subplots()
kr_a = ax1.contour(x1v, x2v, y_act, levels=[0], colors='Black')

# Plot for scattered data
ID1 = ax1.scatter(ak.samples[nd:, 0], ak.samples[nd:, 1], color='Grey', label='New samples')
ID = ax1.scatter(x.samples[:nd, 0], x.samples[:nd, 1], color='Red', label='Initial samples')
plt.legend(handles=[ID1, ID])
plt.show()

# %% md
#
# Monte Carlo Simulation
# -----------------------
# Probability of failure and covariance is estimated using Monte Carlo Simulation. 10,000 samples are generated
# randomly using :class:`.MonteCarloSampling` class and model is evaluated at all samples.

# %%

start_time = time.time()

# Code
b = MonteCarloSampling(distributions=marginals, nsamples=10 ** 4, random_state=4)
model = PythonModel(model_script='local_series.py', model_object_name='series')
r1model = RunModel(model=model)
r1model.run(samples=b.samples)


gx = np.array(r1model.qoi_list)
pf_mcs = np.sum(np.array(gx) < 0) / b.nsamples
cov_pf_mcs = np.sqrt((1 - pf_mcs) / (pf_mcs * b.nsamples))
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

# %% md
#
# Results from Monte Carlo Simulation.

# %%

print('Time: ', elapsed_time)
print('Function evaluation: ', b.nsamples)
print('Probability of failure: ', pf_mcs)

