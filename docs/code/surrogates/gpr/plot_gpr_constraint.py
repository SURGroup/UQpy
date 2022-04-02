"""

Gaussian Process Regressor
======================================================================

"""

# %% md
#
# This jupyter script shows the performance of GaussianProcessRegressor class in the UQpy. A training data is generated
# using a function (:math:`f(x)`, as defined below), which is used to train a surrogate model. The following three cases
# are considered here:
#
# - No Noise
# - Noisy Output
# - Noisy Output with Non-Negative Constraints

# %%

# %% md
#
# Import the necessary modules to run the example script. Notice that FminCobyla is used here, to solve the MLE
# optimization problem with constraints.

# %%

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer
from UQpy.utilities.FminCobyla import FminCobyla
from UQpy.surrogates import GaussianProcessRegressor, Nonnegative, RBF


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
# 1. Train GPR
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

bounds_1 = [[10 ** (-3), 10 ** 3], [10 ** (-3), 10 ** 2]]
optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)

# %% md
#
# Define the 'GaussianProcessRegressor' class object, the input attributes defined here are kernel, optimizer, initial
# estimates of hyperparameters and number of times MLE is identified using random starting point.

# %%

gpr1 = GaussianProcessRegressor(kernel=kernel1, hyperparameters=[10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                optimizations_number=10, noise=False)

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


# %% md
#
# 2. Train GPR
# ~~~~~~~~~~~~~
# - Noise
# - No Constraints
#
# Define kernel used to define the covariance matrix. Here, the application of Radial Basis Function (RBF) kernel is
# demonstrated.

# %%

kernel2 = RBF()


# %% md
#
# Define the optimizer used to identify the maximum likelihood estimate.

# %%

bounds_2 = [[10**(-3), 10**3], [10**(-3), 10**2], [10**(-10), 10**(1)]]
optimizer2 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_2)


# %% md
#
# Define the 'GaussianProcessRegressor' class object, the input attributes defined here are kernel, optimizer, initial
# estimates of hyperparameters and number of times MLE is identified using random starting point.

# %%

gpr2 = GaussianProcessRegressor(kernel=kernel2, hyperparameters=[10**(-3), 10**(-2), 10**(-10)], optimizer=optimizer2,
                                optimizations_number=10, noise=True)


# %% md
#
# Call the 'fit' method to train the surrogate model (GPR).

# %%

gpr2.fit(X_train, y_train)

# %% md
#
# The maximum likelihood estimates of the hyperparameters are as follows:

# %%

print(gpr2.hyperparameters)

print('Length Scale: ', gpr2.hyperparameters[0])
print('Process Variance: ', gpr2.hyperparameters[1])
print('Noise Variance: ', gpr2.hyperparameters[2])


# %% md
#
# Use 'predict' method to compute surrogate prediction at the test samples. The attribute 'return_std' is a boolean
# indicator. If 'True', 'predict' method also returns the standard error at the test samples.

# %%

y_pred2, y_std2 = gpr2.predict(X_test, return_std=True)


# %% md
#
# The plot shows the test function in dashed red line and 13 training points are represented by blue dots. Also, blue
# curve shows the GPR prediction for $x \in (0, 1)$ and yellow shaded region represents 95% confidence interval.

# %%

fig, ax = plt.subplots(figsize=(8.5,7))
ax.plot(X_test,y_test,'r--',linewidth=2,label='Test Function')
ax.plot(X_train,y_train,'bo',markerfacecolor='b', markersize=10, label='Training Data')
ax.plot(X_test,y_pred2,'b-', lw=2, label='GP Prediction')
ax.plot(X_test, np.zeros((X_test.shape[0],1)))
ax.fill_between(X_test.flatten(), y_pred2-1.96*y_std2,
                y_pred2+1.96*y_std2,
                facecolor='yellow',label='95% CI')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('f(x)', fontsize=15)
ax.set_ylim([-0.3,1.8])
plt.title('GP Surrogate (Noise, No Constraints)')
ax.legend(loc="upper right",prop={'size': 12});
plt.grid()

# %% md
#
# 3. Train GPR
# ~~~~~~~~~~~~~
# - Noise
# - Constraints
#
# Here, 30 equidistant point are selected over the domain of :math:`x`, lets call them constraint points. The idea is to
# train the surrogate model such that the probability of positive surrogates prediction is very high at these points.

# %%

X_c = np.linspace(0, 1, 31).reshape(-1,1)
y_c = funct(X_c)

# %% md
#
# In this approach, MLE problem is solved with the following constraints:
#
# .. math:: \hat{y}(x_c)-Z \sigma_{\hat{y}}(x_c) > 0  \quad \quad Z = 2
# .. math:: |\hat{y}(x_t) - y(x_t)| < \epsilon   \quad \quad \epsilon = 0.3
#
# where, :math:`x_c` and :math:`x_t` are the constraint and training sample points, respectively.
#
# Define kernel used to define the covariance matrix. Here, the application of Radial Basis Function (RBF) kernel is
# demonstrated.

# %%

kernel3 = RBF()

# %% md
#
# Define the optimizer used to identify the maximum likelihood estimate.

# %%

bounds_3 = [[10**(-6), 10**(-1)], [10**(-5), 10**(-1)], [10**(-13), 10**(-5)]]
optimizer3 = FminCobyla()

# %% md
#
# Define constraints for the Cobyla optimizer using UQpy's Nonnegatice class.

# %%

cons = Nonnegative(constraint_points=X_c, observed_error=0.03, z_value=2)

# %% md
#
# Define the 'GaussianProcessRegressor' class object, the input attributes defined here are kernel, optimizer, initial
# estimates of hyperparameters and number of times MLE is identified using random starting point.

# %%

gpr3 = GaussianProcessRegressor(kernel=kernel3, hyperparameters=[10**(-3), 10**(-2), 10**(-10)], optimizer=optimizer3,
                                optimizations_number=10, optimize_constraints=cons, bounds=bounds_3, noise=True)

# %% md
#
# Call the 'fit' method to train the surrogate model (GPR).

# %%

gpr3.fit(X_train, y_train)

# %% md
#
# The maximum likelihood estimates of the hyperparameters are as follows:

# %%

print(gpr3.hyperparameters)

print('Length Scale: ', gpr3.hyperparameters[0])
print('Process Variance: ', gpr3.hyperparameters[1])
print('Noise Variance: ', gpr3.hyperparameters[2])


# %% md
#
# Use 'predict' method to compute surrogate prediction at the test samples. The attribute 'return_std' is a boolean
# indicator. If 'True', 'predict' method also returns the standard error at the test samples.

# %%

y_pred3, y_std3 = gpr3.predict(X_test, return_std=True)

# %% md
#
# The plot shows the test function in dashed red line and 13 training points are represented by blue dots. Also, blue
# curve shows the GPR prediction for $x \in (0, 1)$ and yellow shaded region represents 95% confidence interval.

# %%

fig, ax = plt.subplots(figsize=(8.5,7))
ax.plot(X_test,y_test,'r--',linewidth=2,label='Test Function')
ax.plot(X_train,y_train,'bo',markerfacecolor='b', markersize=10, label='Training Data')
ax.plot(X_test,y_pred3,'b-', lw=2, label='GP Prediction')
ax.plot(X_test, np.zeros((X_test.shape[0],1)))
ax.fill_between(X_test.flatten(), y_pred3-1.96*y_std3,
                y_pred3+1.96*y_std3,
                facecolor='yellow',label='95% Credibility Interval')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('f(x)', fontsize=15)
ax.set_ylim([-0.3,1.8])
ax.legend(loc="upper right",prop={'size': 12});
plt.grid()


# %% md
#
# Verify the constraints for the trained surrogate model. Notice that all values are positive, thus constraints are
# satisfied for the constraint points.

# %%

y_, ys_ = gpr3.predict(X_c, return_std=True)
y_ - 2*ys_

# %% md
#
# Notice that all values are negative, thus constraints are satisfied for the training points.

# %%

y_ = gpr3.predict(X_train, return_std=False)
np.abs(y_train[:, 0]-y_) - 0.03
