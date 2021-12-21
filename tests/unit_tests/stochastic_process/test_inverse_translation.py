import numpy as np
from UQpy.distributions import Uniform
from UQpy.stochastic_process import SpectralRepresentation, Translation, InverseTranslation

n_sim = 100  # Num of samples
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = F / nw
w = np.linspace(0, F - dw, nw)
S = 125 / 4 * w ** 2 * np.exp(-5 * w)
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


R = S_to_R(S, w, t)
distribution = Uniform(0, 1)

Translate_object = Translation(distributions=distribution, time_interval=dt, frequency_interval=dw,
                               number_time_intervals=nt, number_frequency_intervals=nw, correlation_function_gaussian=R,
                               samples_gaussian=samples)

samples_ng = Translate_object.samples_non_gaussian
R_ng = Translate_object.scaled_correlation_function_non_gaussian

InverseTranslate_object = InverseTranslation(distributions=distribution, time_interval=dt, frequency_interval=dw,
                                             number_time_intervals=nt, number_frequency_intervals=nw,
                                             correlation_function_non_gaussian=R_ng, samples_non_gaussian=samples_ng,
                                             percentage_error=5.0)
samples_g = InverseTranslate_object.samples_gaussian
S_g = InverseTranslate_object.power_spectrum_gaussian
R_g = InverseTranslate_object.auto_correlation_function_gaussian
r_g = InverseTranslate_object.correlation_function_gaussian


def test_samples_shape():
    assert samples_g.shape == samples_ng.shape


def test_samples_g_value():
    assert np.isclose(samples_g[25, 0, 43], 0.2544126816395569)


def test_R_g_value():
    assert np.isclose(R_g[42], 0.06893298630483506)

