from UQpy.StochasticProcess import KLE
import numpy as np

n_sim = 100  # Num of samples
m = 400 + 1
T = 1000
dt = T / (m - 1)
t = np.linspace(0, T, m)

# Target Covariance(ACF)
R = np.zeros([m, m])
for i in range(m):
    for j in range(m):
        R[i, j] = 2 * np.exp(-((t[j] - t[i]) / 281) ** 2)

KLE_Object = KLE(n_sim, R, dt, verbose=True, random_state=128)
samples = KLE_Object.samples


def test_samples_shape():
    assert samples.shape == (n_sim, 1, len(t))


def test_samples_values():
    assert np.isclose(samples[27, 0, 246], 0.22392952712490516, rtol=0.01)
