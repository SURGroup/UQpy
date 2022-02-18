import numpy as np
import scipy
from line_profiler_pycharm import profile

from UQpy import GrassmannPoint, ProjectionKernel
from UQpy.dimension_reduction.kernels.GaussianKernel import GaussianKernel
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
    dmaps = DiffusionMaps.create_from_data(data=swiss_roll,
                                           alpha=0.5, n_eigenvectors=3,
                                           is_sparse=True, neighbors_number=100,
                                           kernel=GaussianKernel(epsilon=0.03))

    diff_coords, evals, evecs = dmaps.fit()

    assert round(evals[0], 9) == 1.0
    assert round(evals[1], 9) == 0.999489295
    assert round(evals[2], 9) == 0.999116766


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

    dmaps = DiffusionMaps.create_from_data(data=X, alpha=1, n_eigenvectors=3,
                                           kernel=GaussianKernel(epsilon=0.3))

    diff_coords, evals, evecs = dmaps.fit()

    assert evals[0] == 1.0000000000000002
    assert evals[1] == 0.9964842223723996
    assert evals[2] == 0.9964453129642372


def test_diff_matrices():
    import numpy as np

    np.random.seed(111)
    npts = 1000
    pts = np.random.rand(npts, 2)

    a0 = 0
    a1 = 1
    b0 = 0
    b1 = 1

    nodes = np.zeros(np.shape(pts))

    nodes[:, 0] = pts[:, 0] * (a1 - a0) + a0
    nodes[:, 1] = pts[:, 1] * (b1 - b0) + b0

    ns = 40

    x = np.linspace(0, 1, ns)
    samples = []
    for i in range(npts):

        M = np.zeros((ns, ns))
        for k in range(ns):
            f = np.sin(0.1 * k * np.pi * nodes[i, 0] * x + 2 * np.pi * nodes[i, 1])
            M[:, k] = f

        samples.append(M)

    dmaps = DiffusionMaps.create_from_data(data=samples, alpha=0.5, n_eigenvectors=10)

    diff_coords, evals, evecs = dmaps.fit()

    assert evals[0] == 1.0000000000000004
    assert evals[1] == 0.12956887787384186
    assert evals[2] == 0.07277038589085978

