"""

Gaussian Process without noise
======================================================================

"""

# %% md
#
# This jupyter script shows the performance of GaussianProcessRegressor class in the UQpy. A training data is generated
# using a function (:math:`f(x)`, as defined below), which is used to train a surrogate model.

# %%

# %% md
#
# Import the necessary modules to run the example script. Notice that FminCobyla is used here, to solve the MLE
# optimization problem with constraints.

# %%

import warnings

import matplotlib.pyplot as plt
import numpy as np

from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
from UQpy.utilities import RBF

warnings.filterwarnings('ignore')
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer
from UQpy.surrogates import GaussianProcessRegression


# %% md
#
# Consider the following function :math:`f(x)`.
#
# .. math:: f(x) = \frac{1}{100} + \frac{5}{8}(2x-1)^4[(2x-1)^2 + 4\sin{(5 \pi x)^2}], \quad \quad x \in [0,1]

# %%

def funct(x):
    y = (1 / 100) + (5 / 8) * ((2 * x - 1) ** 4) * (((2 * x - 1) ** 2) + 4 * np.sin(5 * np.pi * x) ** 2)
    return y


# %% md
#
# Define the training data set. The following 13 points have been used to fit the GP.

# %%

X_train = np.array([0, 0.06, 0.08, 0.26, 0.27, 0.4, 0.52, 0.6, 0.68, 0.81, 0.9, 0.925, 1]).reshape(-1, 1)
y_train = funct(X_train)

# %% md
#
# Define the test data.

# %%

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = funct(X_test)

# %% md
#
# The plot shows the test function in dashed red line and 13 training points are represented by blue dots.

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X_test, y_test, 'r--', linewidth=2, label='Test Function')
ax.plot(X_train, y_train, 'bo', markerfacecolor='b', markersize=10, label='Training Data')
ax.plot(X_test, np.zeros((X_test.shape[0], 1)))
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('f(x)', fontsize=15)
ax.set_ylim([-0.3, 1.8])
ax.legend(loc="upper right", prop={'size': 12});
plt.grid()

# %% md
#
# Train GPR
# ~~~~~~~~~~~~~
# - No Noise
# - No Constraints
#
# Define kernel used to define the covariance matrix. Here, the application of Radial Basis Function (RBF) kernel is
# demonstrated.

# %%

kernel1 = RBF()

# %% md
#
# Define the optimizer used to identify the maximum likelihood estimate.

# %%

bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2]]
optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)

# %% md
#
# Define the 'GaussianProcessRegressor' class object, the input attributes defined here are kernel, optimizer, initial
# estimates of hyperparameters and number of times MLE is identified using random starting point.

# %%

gpr1 = GaussianProcessRegression(kernel=kernel1, hyperparameters=[10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                 optimizations_number=10, noise=False, regression_model=LinearRegression())

# %% md
#
# Call the 'fit' method to train the surrogate model (GPR).

# %%

gpr1.fit(X_train, y_train)


# %% md
#
# The maximum likelihood estimates of the hyperparameters are as follows:

# %%

gpr1.hyperparameters

print('Length Scale: ', gpr1.hyperparameters[0])
print('Process Variance: ', gpr1.hyperparameters[1])

# %% md
#
# Use 'predict' method to compute surrogate prediction at the test samples. The attribute 'return_std' is a boolean
# indicator. If 'True', 'predict' method also returns the standard error at the test samples.

# %%

y_pred1, y_std1 = gpr1.predict(X_test, return_std=True)


# %% md
#
# The plot shows the test function in dashed red line and 13 training points are represented by blue dots. Also, blue
# curve shows the GPR prediction for :math:`x \in (0, 1)` and yellow shaded region represents 95% confidence interval.

# %%

fig, ax = plt.subplots(figsize=(8.5,7))
ax.plot(X_test,y_test,'r--',linewidth=2,label='Test Function')
ax.plot(X_train,y_train,'bo',markerfacecolor='b', markersize=10, label='Training Data')
ax.plot(X_test,y_pred1,'b-', lw=2, label='GP Prediction')
ax.plot(X_test, np.zeros((X_test.shape[0],1)))
ax.fill_between(X_test.flatten(), y_pred1-1.96*y_std1,
                y_pred1+1.96*y_std1,
                facecolor='yellow',label='95% CI')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('f(x)', fontsize=15)
ax.set_ylim([-0.3,1.8])
plt.title('GP surrogate (No noise, No Constraints)')
ax.legend(loc="upper right",prop={'size': 12})
plt.grid()