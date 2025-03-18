import numpy as np
import scipy.stats as stats
from UQpy.utilities.ValidationTypes import RandomStateType

from UQpy.run_model.RunModel import RunModel


def svd(matrix, rank=None, tol=None):
    """
    Compute the singular value decomposition (SVD) of a matrix.

    **Inputs:**

    * **matrix** (`ndarray`):
        Matrix of ``shape=(m, n)`` to perform the factorization using thin SVD

    * **tol** (`float`):
        Tolerance to estimate the rank of the matrix.

        Default: Machine precision

    * **iterations** (`rank`):
        Number of eigenvalues to keep.

        Default: None

    **Output/Returns:**

    * **u** (`ndarray`):
        Matrix of left eigenvectors of ``shape=(m, rank)``.

    * **v** (`ndarray`):
        Matrix of right eigenvectors of ``shape=(rank, n)``.

    * **s** (`ndarray`):
        Matrix of eigenvalues ``shape=(rank, rank)``.

    """
    ui, si, vi = np.linalg.svd(matrix, full_matrices=True, hermitian=False)
    si = np.diag(si)
    vi = vi.T
    if rank is None:
        if tol is not None:
            rank = np.linalg.matrix_rank(si, tol=tol)
        else:
            rank = np.linalg.matrix_rank(si)
        u = ui[:, :rank]
        s = si[:rank, :rank]
        v = vi[:, :rank]
    else:
        u = ui[:, :rank]
        s = si[:rank, :rank]
        v = vi[:, :rank]

    return u, s, v


def nearest_psd(input_matrix, iterations=10):
    """
    A function to compute the nearest positive semi-definite matrix of a given matrix :cite:`Utilities3`.

    :param numpy.ndarray input_matrix: Matrix to find the nearest PD.
    :param iterations: Number of iterations to perform. Default: 10
    :return: Nearest PSD matrix to input_matrix.
    """
    n = input_matrix.shape[0]
    w = np.identity(n)
    # w is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    delta_s = 0
    psd_matrix = input_matrix.copy()
    for k in range(iterations):

        r_k = psd_matrix - delta_s
        x_k = _get_ps(r_k, w=w)
        delta_s = x_k - r_k
        psd_matrix = _get_pu(x_k, w=w)

    return psd_matrix


def nearest_pd(input_matrix):
    """
    This is a method to find the nearest positive-definite matrix to input (:cite:`Utilities1`, :cite:`nearest_psd`).

    :param numpy.ndarray input_matrix: Matrix to find the nearest PD.
    :return: Nearest PD matrix to input_matrix.
    """
    b = (input_matrix + input_matrix.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))

    a2 = (b + h) / 2

    pd_matrix = (a2 + a2.T) / 2

    if _is_pd(pd_matrix):
        return pd_matrix

    spacing = np.spacing(np.linalg.norm(pd_matrix))
    k = 1
    while not _is_pd(pd_matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(pd_matrix)))
        pd_matrix += np.eye(input_matrix.shape[0]) * (-min_eig * k ** 2 + spacing)
        k += 1

    return pd_matrix


def _is_pd(input_matrix):
    try:
        _ = np.linalg.cholesky(input_matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def run_parallel_python(model_script, model_object_name, sample, dict_kwargs=None):
    """
    Method needed by :class:`.RunModel` to execute a python model in parallel
    """
    _ = sample
    exec("from " + model_script[:-3] + " import " + model_object_name)

    if dict_kwargs is None:
        par_res = eval(model_object_name + "(sample)")
    else:
        par_res = eval(model_object_name + "(sample, **dict_kwargs)")

    return par_res


def gradient(runmodel_object=None, point=None, order="first", df_step=None):
    """
    This method estimates the gradients (1st, 2nd, mixed) of a function using a finite difference scheme in the
    standard normal space. First order gradients are calculated using central finite differences.

    :param runmodel_object: The numerical model. It should be of type :class:`.RunModel` or a
     `callable`.
    :param point: The point to evaluate the gradient with shape ``point``.shape=(1, dimension)
    :param order: Order of the gradient. Available options: 'first', 'second', 'mixed'. Default: 'First'.
    :param df_step: Finite difference step. Default: 0.001.
    :return: Vector of first-order gradients (if order = 'first'). Vector of second-order gradients
     (if order = 'second'). Vector of mixed gradients (if order = 'mixed').
    """
    point = np.atleast_2d(point)

    dimension = point.shape[1]

    if df_step is None:
        df_step = [0.001] * dimension
    elif isinstance(df_step, float):
        df_step = [df_step] * dimension
    elif isinstance(df_step, list):
        if len(df_step) == 1:
            df_step = [df_step[0]] * dimension

    if not callable(runmodel_object) and not isinstance(runmodel_object, RunModel):
        raise RuntimeError("A RunModel object or callable function must be provided as model.")

    def func(m):
        def func_eval(x):
            if isinstance(m, RunModel):
                m.run(samples=x, append_samples=False)
                return np.array(m.qoi_list).flatten()
            else:
                return m(x).flatten()

        return func_eval

    f_eval = func(m=runmodel_object)

    if order.lower() == "first":
        du_dj = np.zeros([point.shape[0], dimension])

        for ii in range(dimension):
            eps_i = df_step[ii]
            u_i1_j = point.copy()
            u_i1_j[:, ii] = u_i1_j[:, ii] + eps_i
            u_1i_j = point.copy()
            u_1i_j[:, ii] = u_1i_j[:, ii] - eps_i

            qoi_plus = f_eval(u_i1_j)
            qoi_minus = f_eval(u_1i_j)

            du_dj[:, ii] = (qoi_plus - qoi_minus) / (2 * eps_i)

        return du_dj

    elif order.lower() == "second":
        # print('Calculating second order derivatives..')
        d2u_dj = np.zeros([point.shape[0], dimension])
        for ii in range(dimension):
            u_i1_j = point.copy()
            u_i1_j[:, ii] = u_i1_j[:, ii] + df_step[ii]
            u_1i_j = point.copy()
            u_1i_j[:, ii] = u_1i_j[:, ii] - df_step[ii]

            qoi_plus = f_eval(u_i1_j)
            qoi = f_eval(point)
            qoi_minus = f_eval(u_1i_j)

            d2u_dj[:, ii] = (qoi_plus - 2 * qoi + qoi_minus) / (
                df_step[ii] * df_step[ii]
            )

        return d2u_dj

    elif order.lower() == "mixed":

        import itertools

        range_ = list(range(dimension))
        d2u_dij = np.zeros([point.shape[0], int(dimension * (dimension - 1) / 2)])
        count = 0
        for i in itertools.combinations(range_, 2):
            u_i1_j1 = point.copy()
            u_i1_1j = point.copy()
            u_1i_j1 = point.copy()
            u_1i_1j = point.copy()

            eps_i1_0 = df_step[i[0]]
            eps_i1_1 = df_step[i[1]]

            u_i1_j1[:, i[0]] += eps_i1_0
            u_i1_j1[:, i[1]] += eps_i1_1

            u_i1_1j[:, i[0]] += eps_i1_0
            u_i1_1j[:, i[1]] -= eps_i1_1

            u_1i_j1[:, i[0]] -= eps_i1_0
            u_1i_j1[:, i[1]] += eps_i1_1

            u_1i_1j[:, i[0]] -= eps_i1_0
            u_1i_1j[:, i[1]] -= eps_i1_1

            print("hi")
            qoi_0 = f_eval(u_i1_j1)
            qoi_1 = f_eval(u_i1_1j)
            qoi_2 = f_eval(u_1i_j1)
            qoi_3 = f_eval(u_1i_1j)

            d2u_dij[:, count] = (qoi_0 + qoi_3 - qoi_1 - qoi_2) / (
                4 * eps_i1_0 * eps_i1_1
            )

            count += 1
        return d2u_dij


def bi_variate_normal_pdf(x1, x2, rho):
    return (
        1
        / (2 * np.pi * np.sqrt(1 - rho ** 2))
        * np.exp(-1 / (2 * (1 - rho ** 2)) * (x1 ** 2 - 2 * rho * x1 * x2 + x2 ** 2))
    )

def _get_a_plus(a):
    eig_val, eig_vec = np.linalg.eig(a)
    q = np.matrix(eig_vec)
    x_diagonal = np.matrix(np.diag(np.maximum(eig_val, 0)))

    return q * x_diagonal * q.T


def _get_ps(a, w=None):
    w05 = np.matrix(w ** 0.5)

    return w05.I * _get_a_plus(w05 * a * w05) * w05.I


def _get_pu(a, w=None):
    a_ret = np.array(a.copy())
    a_ret[w > 0] = np.array(w)[w > 0]
    return np.matrix(a_ret)


def _nn_coord(x, k):
    """
    Select k elements close to x.
    Select k elements close to x to be used to construct a sparse kernel
    matrix to be used in the diffusion maps.

    :param Union[list, numpy.ndarray] x: Matrices to be tested.
    :param int k: Number of points close to x.
    :return: Indices of the closer points.
    :rtype: int
    """
    if isinstance(x, list):
        x = np.array(x)

    dim = np.shape(x)

    if len(dim) != 1:
        raise ValueError("k MUST be a vector.")

    if not isinstance(k, int):
        raise TypeError("k MUST be integer.")

    if k < 1:
        raise ValueError("k MUST be larger than or equal to 1.")

    idx = x.argsort()[: len(x) - k]
    return idx


def correlation_distortion(dist_object, rho):
    """
    This method computes the correlation distortion from Gaussian distribution to any other distribution defined in
    UQpy.distributions

    :param Distribution dist_object: The object of the Distribution the correlation needs to be calculated.
    :param float rho: The Gaussian  correlation value.
    :return: The distorted correlation value.
    :rtype: float
    """

    if np.isclose(rho, 1.0):
        rho = 0.999
    n = 512
    zmax = 8
    zmin = -zmax
    eta, w2d, xi = calculate_gauss_quadrature_2d(n, zmax, zmin)
    tmp_f_xi = dist_object.icdf(stats.norm.cdf(xi[:, np.newaxis]))
    tmp_f_eta = dist_object.icdf(stats.norm.cdf(eta[:, np.newaxis]))
    coef = tmp_f_xi * tmp_f_eta * w2d
    phi2 = bi_variate_normal_pdf(xi, eta, rho)
    rho_non = np.sum(coef * phi2)
    rho_non = (
        rho_non - dist_object.moments(moments2return="m") ** 2
    ) / dist_object.moments(moments2return="v")
    return rho_non


def calculate_gauss_quadrature_2d(n, zmax, zmin):
    points, weights = np.polynomial.legendre.leggauss(n)
    points = -(0.5 * (points + 1) * (zmax - zmin) + zmin)
    weights = weights * (0.5 * (zmax - zmin))
    xi = np.tile(points, [n, 1])
    xi = xi.flatten(order="F")
    eta = np.tile(points, n)
    first = np.tile(weights, n)
    first = np.reshape(first, [n, n])
    second = np.transpose(first)
    weights2d = first * second
    w2d = weights2d.flatten()
    return eta, w2d, xi


def process_random_state(random_state: RandomStateType):
    if isinstance(random_state, (int, type(None))):
        return np.random.RandomState(random_state)
    else:
        return random_state


def calculate_gradient(krig_object, step_size, x, y, xt):
    """
    Estimating gradients with a Kriging metamodel (surrogate).

    :param krig_object:
    :param step_size:
    :param numpy.ndarray x: Samples in the training data.
    :param numpy.ndarray y: Function values evaluated at the samples in the training data.
    :param numpy.ndarray xt: Samples where gradients need to be evaluated.
    :return: First-order gradient evaluated at the points 'xt' using central difference.
    :rtype: numpy.ndarray
    """

    if krig_object is not None:
        krig_object.fit(x, y)
        krig_object.nopt = 1
        tck = krig_object.predict
    else:
        from scipy.interpolate import LinearNDInterpolator

        tck = LinearNDInterpolator(x, y, fill_value=0).__call__

    gr = gradient(point=xt, runmodel_object=tck, order="first", df_step=step_size)
    return gr
