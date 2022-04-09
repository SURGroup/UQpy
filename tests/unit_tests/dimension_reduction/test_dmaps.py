from UQpy.utilities.kernels.GaussianKernel import GaussianKernel
from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps


def test_dmaps_swiss_roll():
    import numpy as np
    from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps

    # set parameters
    length_phi = 15  # length of swiss roll in angular direction
    length_Z = 15  # length of swiss roll in z direction
    sigma = 0.1  # noise strength
    m = 4000  # number of samples

    np.random.seed(1111)
    # create dataset
    phi = length_phi * np.random.rand(m)
    xi = np.random.rand(m)
    Z0 = length_Z * np.random.rand(m)
    X0 = 1. / 6 * (phi + sigma * xi) * np.sin(phi)
    Y0 = 1. / 6 * (phi + sigma * xi) * np.cos(phi)

    swiss_roll = np.array([X0, Y0, Z0]).transpose()
    dmaps = DiffusionMaps(data=swiss_roll, kernel=GaussianKernel(epsilon=0.03),
                          alpha=0.5, n_eigenvectors=3, is_sparse=True, n_neighbors=100)

    evals = dmaps.eigenvalues
    assert round(evals[0], 9) == 1.0
    assert round(evals[1], 9) == 0.999174326
    assert round(evals[2], 9) == 0.998335336


def test_dmaps_circular():
    import numpy as np
    np.random.seed(1111)
    a = 6
    b = 1
    k = 10
    u = np.linspace(0, 2 * np.pi, 1000)

    v = k * u

    x0 = (a + b * np.cos(0.8 * v)) * (np.cos(u))
    y0 = (a + b * np.cos(0.8 * v)) * (np.sin(u))
    z0 = b * np.sin(0.8 * v)

    rox = 0.2
    roy = 0.2
    roz = 0.2
    x = x0 + rox * np.random.normal(0, 1, len(x0))
    y = y0 + roy * np.random.normal(0, 1, len(y0))
    z = z0 + roz * np.random.normal(0, 1, len(z0))

    X = np.array([x, y, z]).transpose()

    dmaps = DiffusionMaps(data=X, alpha=1, n_eigenvectors=3,
                          kernel=GaussianKernel(epsilon=0.3))

    evals = dmaps.eigenvalues
    assert np.round(evals[0], 5) == 1.0
    assert np.round(evals[1], 5) == 0.99826
    assert np.round(evals[2], 5) == 0.99824
