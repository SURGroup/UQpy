from UQpy.StochasticProcess import SRM
import numpy as np


n_sim = 100  # Num of samples
n = 1  # Num of dimensions
m = 1  # Num of variables
T = 100  # Time(1 / T = dw)
nt = 256  # Num of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = F / nw
w = np.linspace(0, F - dw, nw)
t_u = 2 * np.pi / 2 / F

S_1d_1v = 125 / 4 * w ** 2 * np.exp(-5 * w)
SRM_object = SRM(n_sim, S_1d_1v, dt, dw, nt, nw, verbose=True, random_state=128)
samples_1d_1v = SRM_object.samples


def test_samples_1d_1v_shape():
    assert samples_1d_1v.shape == (n_sim, 1, nt)


def test_samples_1d_1v_value():
    assert np.isclose(samples_1d_1v[53, 0, 134], -0.9143690244714813)
