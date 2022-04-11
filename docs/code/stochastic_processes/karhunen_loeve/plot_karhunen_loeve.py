"""

Karhunen Loeve Expansion
=================================================================

In this example, the KL Expansion is used to generate stochastic processes from a prescribed Autocorrelation Function.
This example illustrates how to use the :class:`.KarhunenLoeveExpansion` class for a one dimensional and compare the
statistics of the generated stochastic processes with the expected values.

"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.KarhunenLoeveExpansion` class from the :py:mod:`stochastic_processes` module of UQpy.

#%%

from UQpy.stochastic_process import KarhunenLoeveExpansion
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%% md
#
# The input parameters necessary for the generation of the stochastic processes are given below:

#%%

n_sim = 10000  # Num of samples

m = 400 + 1
T = 1000
dt = T / (m - 1)
t = np.linspace(0, T, m)

#%% md
#
# Defining the Autocorrelation Function.

#%%

# Target Covariance(ACF)
R = np.zeros([m, m])
for i in range(m):
    for j in range(m):
        R[i, j] = 2 * np.exp(-((t[j] - t[i]) / 281) ** 2)  # var = 2

KLE_Object = KarhunenLoeveExpansion(n_samples=n_sim, correlation_function=R, time_interval=dt)
samples = KLE_Object.samples

fig, ax = plt.subplots()
plt.title('Realisation of the Karhunen Loeve Expansion')
plt.plot(t, samples[0, 0])
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()

print('The mean of the samples is ', np.mean(samples), 'whereas the expected mean is 0.000')
print('The variance of the samples is ', np.var(samples), 'whereas the expected variance is 2.000')