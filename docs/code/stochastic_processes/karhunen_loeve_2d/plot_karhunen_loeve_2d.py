"""

Karhunen Loeve Expansion 2 Dimesional
=================================================================

In this example, the KL Expansion is used to generate 2 dimensional stochastic fields from a prescribed Autocorrelation
Function. This example illustrates how to use the :class:`.KarhunenLoeveExpansion2D` class for a 2 dimensional
random field and compare the statistics of the generated random field with the expected values.

"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.KarhunenLoeveExpansionTwoDimension` class from the :py:mod:`stochastic_processes` module of UQpy.

# %%
from matplotlib import pyplot as plt
from UQpy.stochastic_process import KarhunenLoeveExpansion2D
import numpy as np


# %% md
#
# The input parameters necessary for the generation of the stochastic processes are given below:

# %%

n_samples = 100  # Num of samples
nx, nt = 20, 10
dx, dt = 0.05, 0.1

x = np.linspace(0, (nx - 1) * dx, nx)
t = np.linspace(0, (nt - 1) * dt, nt)
xt_list = np.meshgrid(x, x, t, t, indexing='ij')  # R(t_1, t_2, x_1, x_2)

# %% md

# Defining the Autocorrelation Function.

# %%

R = np.exp(-(xt_list[0] - xt_list[1]) ** 2 - (xt_list[2] - xt_list[3]) ** 2)
# R(x_1, x_2, t_1, t_2) = exp(-(x_1 - x_2) ** 2 -(t_1 - t_2) ** 2)

KLE_Object = KarhunenLoeveExpansion2D(n_samples=n_samples, correlation_function=R, time_intervals=np.array([dt, dx]),
                                      thresholds=[4, 5], random_state=128)

# %% md

# Simulating the samples.

# %%

samples = KLE_Object.samples

# %% md

# Plotting a sample of the stochastic field.

# %%

fig = plt.figure()
plt.title('Realisation of the Karhunen Loeve Expansion for a 2D stochastic field')
plt.imshow(samples[0, 0])
plt.ylabel('t (Time)')
plt.xlabel('x (Space)')
plt.show()

print('The mean of the samples is ', np.mean(samples), 'whereas the expected mean is 0.000')
print('The variance of the samples is ', np.var(samples), 'whereas the expected variance is 1.000')
