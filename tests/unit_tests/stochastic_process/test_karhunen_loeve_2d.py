from UQpy.stochastic_process.KarhunenLoeveExpansion2D import KarhunenLoeveExpansion2D
import numpy as np

n_samples = 100_000  # Num of samples
nx, nt = 20, 10
dx, dt = 0.05, 0.1

x = np.linspace(0, (nx - 1) * dx, nx)
t = np.linspace(0, (nt - 1) * dt, nt)
xt_list = np.meshgrid(x, x, t, t, indexing='ij')  # R(t_1, t_2, x_1, x_2)

R = np.exp(-(xt_list[0] - xt_list[1]) ** 2 - (xt_list[2] - xt_list[3]) ** 2)
# R(x_1, x_2, t_1, t_2) = exp(-(x_1 - x_2) ** 2 -(t_1 - t_2) ** 2)

KLE_Object = KarhunenLoeveExpansion2D(n_samples=n_samples, correlation_function=R,
                                      time_intervals=np.array([dt, dx]), thresholds=[4, 5], random_state=128)
samples = KLE_Object.samples


def test_samples_shape():
    assert samples.shape == (n_samples, 1, len(x), len(t))


def test_samples_values():
    assert np.isclose(samples[13, 0, 13, 6], -1.8616264088843137, rtol=0.01)


def test_run_method():
    nsamples_second_run = 100_000
    KLE_Object.run(n_samples=nsamples_second_run)
    samples = KLE_Object.samples
    assert samples.shape == (n_samples + nsamples_second_run, 1, len(x), len(t))


def test_empirical_correlation():
    assert np.isclose(np.mean(samples[:, 0, 5, 6] * samples[:, 0, 3, 2]), R[5, 3, 6, 2], rtol=0.01)
    assert np.isclose(np.mean(samples[:, 0, 13, 3] * samples[:, 0, 8, 4]), R[13, 8, 3, 4], rtol=0.01)
