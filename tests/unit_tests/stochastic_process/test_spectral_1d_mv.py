from UQpy.stochastic_process import SpectralRepresentation
import numpy as np

np.random.RandomState(1024)

n_sim = 100  # Num of samples
n = 1  # Num of dimensions
m = 3  # Num of variables

T = 10  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

t_u = 2 * np.pi / 2 / F

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
S_1d_mv = S_jk * g_jk

SRM_object = SpectralRepresentation(n_sim, S_1d_mv, dt, df, nt, nf, random_state=128)
samples_1d_mv = SRM_object.samples


def test_samples_1d_mv_shape():
    assert samples_1d_mv.shape == (n_sim, m, nt)


# def test_samples_1d_mv_values():
#     assert np.isclose(-6.292191903354104, samples_1d_mv[43, 2, 67])
