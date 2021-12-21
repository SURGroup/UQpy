from UQpy.stochastic_process import BispectralRepresentation
import numpy as np

n_sim = 100  # Num of samples
n = 1  # Num of dimensions
# Input parameters
T = 60  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.
# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

S = 32 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)
# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(f, f)

b = 95 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = b
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

BSRM_object = BispectralRepresentation(n_sim, S, B_Complex, [dt], [df], [nt], [nf], random_state=128)
samples_1d = BSRM_object.samples


def test_samples_1d_shape():
    assert samples_1d.shape == (n_sim, 1, nt)


def test_samples_1d_values():
    assert (samples_1d[65, 0, 42], -4.203128646610985)