"""

One-dimensional & multiple variables
=================================================================

In this example, the Spectral Representation Method is used to generate stochastic processes from a prescribed Power
Spectrum and associated Cross Spectral Density. This example illustrates how to use the SRM class for a one dimensional
and 'm' variable case and compare the statistics of the generated stochastic processes with the expected values.

"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.SpectralRepresentation` class from the :class:`stochastic_processes` module of UQpy.

#%%

from UQpy.stochastic_process import SpectralRepresentation
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%% md
#
# The input parameters necessary for the generation of the stochastic processes are given below:

#%%

n_sim = 10000  # Num of samples

n = 1 # Num of dimensions

m = 3 # Num of variables

T = 10  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

#%% md
#
# Make sure that the input parameters are in order to prevent aliasing

#%%

t_u = 2*np.pi/2/F

if dt>t_u:
    print('Error')

#%% md
#
# Defining the Power Spectral Density Function (S) and the Cross Spectral Density (g)

#%%

S_11 = 38.3 / (1 + 6.19 * f) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * f) ** (5 / 3)
S_33 = 135 / (1 + 21.8 * f) ** (5 / 3)

g_12 = np.exp(-0.1757 * f)
g_13 = np.exp(-3.478 * f)
g_23 = np.exp(-3.392 * f)

S_list = np.array([S_11, S_22, S_33])
g_list = np.array([g_12, g_13, g_23])

# Assembly of S_jk
S_sqrt = np.sqrt(S_list)
S_jk = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)
# Assembly of g_jk
g_jk = np.zeros_like(S_jk)
counter = 0
for i in range(m):
    for j in range(i + 1, m):
        g_jk[i, j] = g_list[counter]
        counter = counter + 1
g_jk = np.einsum('ij...->ji...', g_jk) + g_jk

for i in range(m):
    g_jk[i, i] = np.ones_like(S_jk[0, 0])
S = S_jk * g_jk


SRM_object = SpectralRepresentation(n_sim, S, dt, df, nt, nf)
samples = SRM_object.samples

fig, ax = plt.subplots()
plt.title('Realisation of the Spectral Representation Method')
plt.plot(t, samples[0, 0], label='1st dimension')
plt.plot(t, samples[0, 1], label='2nd dimension')
plt.plot(t, samples[0, 2], label='3rd dimension')
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.legend()
plt.show()

print('The mean of the samples is ', np.mean(samples), 'whereas the expected mean is 0.000')
print('The variance of the samples is ', np.var(samples), 'whereas the expected variance is ',
      np.sum(S_list)*np.prod(df)*2/m)