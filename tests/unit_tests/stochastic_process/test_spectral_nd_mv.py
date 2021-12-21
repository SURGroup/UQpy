from UQpy.stochastic_process import SpectralRepresentation
import numpy as np

n_sim = 10  # Number of samples
n = 2  # Number of Dimensions
m = 3  # Number of Variables

# Time
T = 10  # Simulation Time
dt = 0.1
nt = int(T / dt) + 1
t = np.linspace(0, T, nt)

# Frequency
nw = 100
W = np.array([1.5, 2.5])
dw = W / (nw - 1)

x_list = [np.linspace(dw[i], W[i], nw) for i in range(n)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))

S_11 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))
S_22 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-3 * np.linalg.norm(xy_list, axis=0))
S_33 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-7 * np.linalg.norm(xy_list, axis=0))

g_12 = np.exp(-0.1757 * np.linalg.norm(xy_list, axis=0))
g_13 = np.exp(-3.478 * np.linalg.norm(xy_list, axis=0))
g_23 = np.exp(-3.392 * np.linalg.norm(xy_list, axis=0))

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
S_nd_mv = S_jk * g_jk

SRM_object = SpectralRepresentation(n_sim, S_nd_mv, [dt, dt], dw, [nt, nt], [nw, nw], random_state=128)
samples_nd_mv = SRM_object.samples


def test_samples_nd_mv_shape():
    assert samples_nd_mv.shape == (n_sim, m, nt, nt)


def test_samples_nd_mv_values():
    assert np.isclose(samples_nd_mv[3, 1, 31, 79], 0.7922504882569233)
