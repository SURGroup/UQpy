"""

Translation
=================================================================

In this example, a Gaussian stochastic processes is translated into a stochastic processes of a number of distributions.
This example illustrates how to use the Translate class to translate from Gaussian to other probability distributions
and compare how the statistics of the translated stochastic processes change along with distributions.

"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the Translate class from the StochasticProcesses module of UQpy.

#%%

from UQpy.stochastic_process import Translation, SpectralRepresentation
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%% md
#
# Firstly we generate Gaussian Stochastic Processes using the Spectral Representation Method.

#%%

n_sim = 10000  # Num of samples
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = F / nw
w = np.linspace(0, F - dw, nw)
S = 125 * w ** 2 * np.exp(-2 * w)

SRM_object = SpectralRepresentation(n_sim, S, dt, dw, nt, nw, random_state=128)
samples = SRM_object.samples

def S_to_R(S, w, t):
    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    R = np.zeros(len(t))
    for i in range(len(t)):
        R[i] = 2 * np.dot(fac, S * np.cos(w * t[i]))
    return R

R_g = S_to_R(S, w, t)
r_g = R_g/R_g[0]

#%% md
#
# We translate the samples to be Uniform samples from 1 to 2

#%%

from UQpy.distributions import Lognormal

distribution = Lognormal(0.5)
samples = samples.flatten()[:, np.newaxis]

Translate_object = Translation(distributions=distribution, time_interval=dt, frequency_interval=dw,
                               number_time_intervals=nt, number_frequency_intervals=nw,
                               correlation_function_gaussian=R_g, samples_gaussian=samples)
samples_ng = Translate_object.samples_non_gaussian

R_ng = Translate_object.scaled_correlation_function_non_gaussian
r_ng = Translate_object.correlation_function_non_gaussian

#%% md
#
# Plotting the actual and translated autocorrelation functions

#%%

fig1 = plt.figure()
plt.plot(r_g, label='Gaussian')
plt.plot(r_ng, label='non-Gaussian')
plt.title('Correlation Function (r)')
plt.legend()
plt.show()