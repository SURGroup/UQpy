import numpy as np
from UQpy import Translation, SRM
from UQpy.Distributions import Uniform

n_sim = 100  # Num of samples
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = F / nw
w = np.linspace(0, F - dw, nw)
S = 125 * w ** 2 * np.exp(-5 * w)
SRM_object = SRM(n_sim, S, dt, dw, nt, nw, verbose=False, random_state=128)
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

distribution = Uniform(0, 1)

Translate_object = Translation(dist_object=distribution, time_interval=dt, frequency_interval=dw,
                               number_time_intervals=nt, number_frequency_intervals=nw,
                               correlation_function_gaussian=R_g, samples_gaussian=samples)
samples_ng = Translate_object.samples_non_gaussian


def test_samples_ng_shape():
    assert samples_ng.shape == samples.shape


def test_samples_ng_values():
    assert np.isclose(samples_ng[18, 0, 153], 0.12719431113548574)


def test_correlation_values():
    assert np.isclose(Translate_object.correlation_function_non_gaussian[52], -0.00604468129285296)

