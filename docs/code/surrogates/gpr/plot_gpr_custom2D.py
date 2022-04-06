"""

Gaussian Process of a custom 2D function
======================================================================

"""

# %% md
#
# In this example, Gaussian Process Regression is used to generate a surrogate model for a given data. In this data,
# sample points are generated using TrueStratifiedSampling class and functional value at sample points are estimated
# using a model defined in python script ('python_model_function.py).

# %%

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the TrueStratifiedSampling, RunModel and GaussianProcessRegression class from UQpy.

# %%

import shutil

from UQpy import PythonModel
from UQpy.surrogates.gaussian_process.regression_models import ConstantRegression
from UQpy.sampling import RectangularStrata
from UQpy.sampling import TrueStratifiedSampling
from UQpy.run_model.RunModel_New import RunModel_New
from UQpy.distributions import Uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from UQpy.surrogates import GaussianProcessRegression, Matern

# %% md
#
# Create a distribution object.

# %%

marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]

# %% md
#
# Create a strata object.

# %%

strata = RectangularStrata(strata_number=[10, 10])

# %% md
#
# Using UQpy TrueStratifiedSampling class to generate samples for two random variables,
# which are uniformly distributed between 0 and 1.

# %%

x = TrueStratifiedSampling(distributions=marginals, strata_object=strata,
                           nsamples_per_stratum=1, random_state=1)

# %% md
#
# RunModel is used to evaluate function values at sample points. Model is defined as a function in python file
# 'python_model_function.py'.

# %%

model = PythonModel(model_script='local_python_model_function.py', model_object_name="y_func")
rmodel = RunModel_New(model=model)

rmodel.run(samples=x.samples)

# %% md
#
# Using UQpy GaussianProcessRegression class to generate a surrogate for generated data. In this illustration, Quadratic regression model and
# Exponential correlation model are used.

# %%

regression_model = ConstantRegression()
kernel = Matern(nu=0.5)

from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

optimizer = MinimizeOptimizer(method="L-BFGS-B")
K = GaussianProcessRegression(regression_model=regression_model, optimizer=optimizer,
                              kernel=kernel,
                              optimizations_number=20,
                              hyperparameters=[1, 1, 0.1])
K.fit(samples=x.samples, values=rmodel.qoi_list)
print(K.hyperparameters)

# %% md
#
# This plot shows the actual model which is used to evaluate the samples to identify the function values.

# %%

num = 25
x1 = np.linspace(0, 1, num)
x2 = np.linspace(0, 1, num)

x1g, x2g = np.meshgrid(x1, x2)
x1gv, x2gv = x1g.reshape(x1g.size, 1), x2g.reshape(x2g.size, 1)

y2 = K.predict(np.concatenate([x1gv, x2gv], 1)).reshape(x1g.shape[0], x1g.shape[1])
model = PythonModel(model_script='local_python_model_function.py', model_object_name="y_func")
r2model = RunModel_New(model=model)
r2model.run(samples=np.concatenate([x1gv, x2gv], 1))
y_act = np.array(r2model.qoi_list).reshape(x1g.shape[0], x1g.shape[1])

fig1 = plt.figure()
ax = fig1.gca(projection='3d')
surf = ax.plot_surface(x1g, x2g, y_act, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-1, 15)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig1.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% md
#
# This plot shows the input data as red dot and green wireframe plot represent the kriging surrogate generated through
# Kriging class.

# %%

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
# Plot for estimated values
kr = ax2.plot_wireframe(x1g, x2g, y2, color='Green', label='Kriging interpolate')

# Plot for scattered data
ID = ax2.scatter3D(x.samples[:, 0], x.samples[:, 1], np.array(rmodel.qoi_list), color='Red', label='Input data')
plt.legend(handles=[kr, ID])
plt.show()