from UQpy.stochastic_process import KarhunenLoeveExpansionTwoDimension
import numpy as np

n_samples = 100  # Num of samples
nx, nt = 20, 10
dx, dt = 0.05, 0.1

x = np.linspace(0, (nx - 1) * dx, nx)
t = np.linspace(0, (nt - 1) * dt, nt)
xt_list = np.meshgrid(x, x, t, t, indexing='ij')  # R(t_1, t_2, x_1, x_2)

R = np.exp(-(xt_list[0] - xt_list[1]) ** 2 - (xt_list[2] - xt_list[3]) ** 2)
# R(x_1, x_2, t_1, t_2) = exp(-(x_1 - x_2) ** 2 -(t_1 - t_2) ** 2)

KLE_Object = KarhunenLoeveExpansionTwoDimension(n_samples=n_samples, correlation_function=R,
                                                time_interval=np.array([dt, dx]), thresholds=[4, 5], random_state=128)
samples = KLE_Object.samples


def test_samples_shape():
    assert samples.shape == (n_samples, 1, len(x), len(t))

def test_samples_values():
    print(samples[13, 0, 13, 6])
    assert np.isclose(samples[13, 0, 13, 6], 0.22392952712490516, rtol=0.01)
