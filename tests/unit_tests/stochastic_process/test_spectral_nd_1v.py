from UQpy.stochastic_process import SpectralRepresentation
import numpy as np

n_sim = 10  # Num of samples
n = 2  # Num of dimensions
m = 1  # Num of variables
T = 10
nt = 200
dt = T / nt
t = np.linspace(0, T - dt, nt)
# Frequency
W = np.array([1.0, 1.5])
nw = 100
dw = W / nw
x_list = [np.linspace(0, W[i] - dw[i], nw) for i in range(n)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))

S_nd_1v = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))
SRM_object = SpectralRepresentation(n_sim, S_nd_1v, [dt, dt], dw, [nt, nt], [nw, nw], random_state=128)
samples_nd_1v = SRM_object.samples


def test_samples_nd_1v_shape():
    assert samples_nd_1v.shape == (n_sim, 1, nt, nt)


# def test_samples_nd_1v_values():
#     assert np.isclose(1.0430071116540038, samples_nd_1v[4, 0, 107, 59])