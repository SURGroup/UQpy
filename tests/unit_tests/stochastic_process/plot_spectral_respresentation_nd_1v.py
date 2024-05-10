import numpy as np
import matplotlib.pyplot as plt
from UQpy.stochastic_process import SpectralRepresentation
plt.style.use('ggplot')


# Sample parameters
n_samples = 1
n_variables = 1

# Spectral Representation Parameters
n_dimension_intervals = np.array([128, 128])
n_dimensions = len(n_dimension_intervals)

n_frequencies = np.array([64, 64])
max_frequency = np.array([6.4, 3.2])

frequency_interval = max_frequency / n_frequencies
frequency_vectors = [np.linspace(0, (n_frequencies[i] - 1) * frequency_interval[i], num=n_frequencies[i])
                     for i in range(n_dimensions)]
# frequencies = np.meshgrid(*frequency_vectors, indexing='ij')

max_time = 2 * np.pi / frequency_interval
dimension_interval = max_time / n_dimension_intervals
x_vector = np.linspace(0, (n_dimension_intervals[0] - 1) * dimension_interval[0], num=n_dimension_intervals[0])
y_vector = np.linspace(0, (n_dimension_intervals[1] - 1) * dimension_interval[1], num=n_dimension_intervals[1])
x_array, y_array = np.meshgrid(x_vector, y_vector)

size = (n_samples, n_frequencies[0], n_frequencies[1])
phi = np.zeros(size)
# phi = np.random.uniform(0, 2 * np.pi, size=size)


def power_spectrum(w_1, w_2):
    """Define n-dimension univariate power spectrum"""
    '''
    works for 
    (0.5, 0) and (-0.5, -0)
    (0, 0.1) and (-0, -0.1)
    (0.5, 0.1) and (-0.5, -0.1)
    '''
    if (w_1, w_2) == (0.5, 0) or (w_1, w_2) == (-0.5, 0):
        return 1
    elif (w_1, w_2) == (0, 0.1) or (w_1, w_2) == (0, -0.1):
        return 1
    elif (w_1, w_2) == (0.5, 0.1) or (w_1, w_2) == (-0.5, -0.1):
        return 1
    else:
        return 0


spectrum = np.full((n_frequencies[0], n_frequencies[1]), np.nan)
for i in range(n_frequencies[0]):
    for j in range(n_frequencies[1]):
        w_1 = i * frequency_interval[0]
        w_2 = j * frequency_interval[1]
        spectrum[i, j] = power_spectrum(w_1, w_2)


def sum_of_cosines(phi):
    """Stochastic process defined using sum of cosines from Eq 56 of
    Simulation of multidimensional Gaussian stochastic fields by spectral representation (1996)"""
    total = np.zeros_like(x_array)
    for i in range(n_frequencies[0]):
        for j in range(n_frequencies[1]):
            kappa_1 = i * frequency_interval[0]
            kappa_2 = j * frequency_interval[1]
            coefficient_1 = np.sqrt(2 * power_spectrum(kappa_1, kappa_2) * np.prod(frequency_interval))
            coefficient_2 = np.sqrt(2 * power_spectrum(kappa_1, -kappa_2) * np.prod(frequency_interval))
            input_1 = (kappa_1 * x_array) + (kappa_2 * y_array) + phi[i, j]
            input_2 = (kappa_1 * x_array) - (kappa_2 * y_array) + phi[i, j]
            term = coefficient_1 * np.cos(input_1) + coefficient_2 * np.cos(input_2)
            total += term
    return np.sqrt(2) * total


def spectral_representation():
    return SpectralRepresentation(spectrum,
                                  dimension_interval,
                                  frequency_interval,
                                  n_dimension_intervals,
                                  n_frequencies)

'''Plot power spectrum'''
# frequency_x, frequency_y = np.meshgrid(*frequency_vectors, indexing='ij')
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(frequency_x, frequency_y, spectrum, cmap='plasma')
# ax.set_title('Power Spectrum $S(\omega_1, \omega_2)$')


srm_object = spectral_representation()
srm_object.run(n_samples, phase_angles=phi)
print('n_variables', srm_object.n_variables)
'''Plot phase angles'''
# frequency_x, frequency_y = np.meshgrid(*frequency_vectors, indexing='ij')
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(frequency_x, frequency_y, srm_object.phi[0, :, :], cmap='plasma')
# ax.set_title('Phase Angles $\Phi$')

'''Plot spectral representation samples'''
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots()
ax.pcolormesh(x_array, y_array, srm_object.samples[0, 0, :, :], vmin=-0.3, vmax=0.7, cmap='plasma')
ax.set_title('Sample of SRM Stochastic Process')
ax.set_aspect('equal')

'''Plot Cosine representation'''
cosines = sum_of_cosines(srm_object.phase_angles[0, :, :])
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots()
ax.pcolormesh(x_array, y_array, cosines, vmin=-0.3, vmax=0.7, cmap='plasma')
# ax.plot_surface(x_array, y_array, cosines, vmin=-0.3, vmax=0.7, cmap='plasma')
ax.set_title('Sample of Cosine Stochastic Process')
ax.set_aspect('equal')
print('all isclose:', (np.isclose(srm_object.samples[0, 0, :, :], cosines)).all())
print('same min:', np.min(srm_object.samples[0, 0, :, :]) == np.min(cosines))
print('same max:', np.max(srm_object.samples[0, 0, :, :]) == np.max(cosines))
print('SRM Mean:', np.mean(srm_object.samples[0, 0, :, :]), 'Cosine Mean:', np.mean(cosines))
plt.show()
