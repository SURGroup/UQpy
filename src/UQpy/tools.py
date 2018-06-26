import numpy as np
import scipy.stats as stats


def transform_z_to_x(samples, marginal_dist, marginal_params):
    samples_transf = np.zeros_like(samples)
    for j in range(len(marginal_dist)):
        icdf = marginal_dist[j].icdf
        for i in range(samples.shape[0]):
            samples_transf[i, j] = icdf(stats.norm.cdf(samples[i, j]), marginal_params[j])
    return samples_transf

def transform_x_to_z(samples, marginal_dist, marginal_params):
    samples_transf = np.zeros_like(samples)
    for j in range(len(marginal_dist)):
        cdf = marginal_dist[j].cdf
        for i in range(samples.shape[0]):
            samples_transf[i, j] = stats.norm.ppf(cdf(samples[i, j], marginal_params[j]))
    return samples_transf


def run_corr(samples, corr):
    """
        A function which performs the Cholesky decomposition of the correlation matrix and correlate
        the samples
    """

    from scipy.linalg import cholesky
    c = cholesky(corr, lower=True)
    y = np.dot(c, samples.T)
    return y.T


def run_uncorr(samples, corr):
    """
        A function which un-correlates
        the samples
    """

    from scipy.linalg import cholesky
    c = cholesky(corr, lower=True)
    inv_corr = np.linalg.inv(c)
    y = np.dot(inv_corr, samples.T)
    return y.T


def solve_double_integral(marginal, params, rho_norm):
    """
        A function to solve the double integral equation in order to evaluate the modified correlation
        matrix in the standard normal space given the correlation matrix in the original space. This is achieved
        by a quadratic two-dimensional Gauss-Legendre integration.
    """

    n = 1024
    zmax = 8
    zmin = -zmax
    points, weights = np.polynomial.legendre.leggauss(n)
    points = - (0.5 * (points + 1) * (zmax - zmin) + zmin)
    weights = weights * (0.5 * (zmax - zmin))

    xi = np.tile(points, [n, 1])
    xi = xi.flatten(order='F')
    eta = np.tile(points, n)

    first = np.tile(weights, n)
    first = np.reshape(first, [n, n])
    second = np.transpose(first)

    weights2d = first * second
    w2d = weights2d.flatten()
    rho = np.ones_like(rho_norm)

    for i in range(len(marginal)):
        icdf_i = marginal[i].icdf
        moments_i = marginal[i].moments
        mi = moments_i(params[i])
        if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
            raise RuntimeError("The marginal distributions need to have "
                               "finite mean and variance")

        for j in range(i + 1, len(marginal)):
            icdf_j = marginal[j].icdf
            moments_j = marginal[j].moments
            mj = moments_j(params[j])
            if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                raise RuntimeError("The marginal distributions need to have "
                                   "finite mean and variance")

            if marginal[j].name == 'Normal' and marginal[i].name == 'Normal':
                rho[i, j] = rho_norm[i, j]
                rho[j, i] = rho[i, j]
            else:
                #  performing Nataf
                tmp_f_xi = ((icdf_j(stats.norm.cdf(xi), params[j]) - mj[0]) / mj[1])
                tmp_f_eta = ((icdf_i(stats.norm.cdf(eta), params[i]) - mi[0]) / mi[1])
                coef = tmp_f_xi * tmp_f_eta * w2d

                rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
                rho[j, i] = rho[i, j]

    return rho


def itam(marginal, params, corr):
    # Initial Guess
    corr_norm0 = corr
    # Iteration Condition
    i_converge = 0
    error0 = 100
    max_iter = 5

    for ii in range(max_iter):
        corr0 = solve_double_integral(marginal, params, corr_norm0)
        # compute the relative difference between the computed NGACF & the target R(Normalized)
        err1 = 0
        err2 = 0
        for i in range(corr0.shape[0]):
            for j in range(corr0.shape[1]):
                err1 = err1 + (corr[i, j] - corr0[i, j]) ** 2
                err2 = err2 + corr0[i, j] ** 2
        error1 = 100 * np.sqrt(err1 / err2)

        if abs(error0 - error1) / error1 < 0.001 or ii == max_iter or 100 * np.sqrt(err1 / err2) < 0.0005:
            i_converge = 1

        corr_norm1 = np.zeros_like(corr_norm0)
        for i in range(corr_norm0.shape[0]):
            for j in range(corr_norm0.shape[1]):
                if corr0[i, j] != 0:
                    corr_norm1[i, j] = (corr[i, j] / corr0[i, j]) * corr_norm0[i, j]
                else:
                    corr_norm1[i, j] = 0

        # Eliminate Numerical error of Upgrading Scheme
        corr_norm1[corr_norm1 < -1.0] = -0.99999
        corr_norm1[corr_norm1 > 1.0] = 0.99999

        # Iteratively finding the nearest PSD(Qi & Sun, 2006)
        corr_norm1 = np.array(near_pd(corr_norm1))

        if i_converge == 0 and ii != max_iter:
            corr_norm0 = corr_norm1
            error0 = error1

    return corr_norm1


def bi_variate_normal_pdf(x1, x2, rho):
    """
        A function which evaluates the values of the bi-variate normal probability distribution function
    """
    return (1 / (2 * np.pi * np.sqrt(1-rho**2)) *
            np.exp(-1/(2*(1-rho**2)) *
                   (x1**2 - 2 * rho * x1 * x2 + x2**2)))


def _get_a_plus(a):

    eig_val, eig_vec = np.linalg.eig(a)
    q = np.matrix(eig_vec)
    xdiag = np.matrix(np.diag(np.maximum(eig_val, 0)))

    return q * xdiag * q.T


def _get_ps(a, w=None):
    w05 = np.matrix(w ** .5)
    return w05.I * _get_a_plus(w05 * a * w05) * w05.I


def _get_pu(a, w=None):

    a_ret = np.array(a.copy())
    a_ret[w > 0] = np.array(w)[w > 0]
    return np.matrix(a_ret)


def near_pd(a, nit=10):

    n = a.shape[0]
    w = np.identity(n)
    # w is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    delta_s = 0
    y_k = a.copy()
    for k in range(nit):
        r_k = y_k - delta_s
        x_k = _get_ps(r_k, w=w)
        delta_s = x_k - r_k
        y_k = _get_pu(x_k, w=w)
    return y_k


def nearest_pd(a):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    b = (a + a.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))

    a2 = (b + h) / 2

    a3 = (a2 + a2.T) / 2

    if is_pd(a3):
        return a3

    spacing = np.spacing(np.linalg.norm(a))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrices with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    k = 1
    while not is_pd(a3):
        min_eig = np.min(np.real(np.linalg.eigvals(a3)))
        a3 += np.eye(a.shape[0]) * (-min_eig * k**2 + spacing)
        k += 1

    return a3


def is_pd(b):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(b)
        return True
    except np.linalg.LinAlgError:
        return False


def estimate_psd(samples, nt, t):
    sample_size = nt
    sample_max_time = t
    dt = t / (nt - 1)
    x_w = np.fft.fft(samples, sample_size, axis=1)
    x_w = x_w[:, 0: int(sample_size / 2)]
    m_ps = np.mean(np.absolute(x_w) ** 2 * sample_max_time / sample_size ** 2, axis=0)
    num = int(t / (2 * dt))
    return np.linspace(0, (1 / (2 * dt) - 1 / t), num), m_ps


def s_to_r(s, w, t):
    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    r = np.zeros([s.shape[0], len(t)])
    for i in range(s.shape[0]):
        for j in range(len(t)):
            if s.shape[0] == 1:
                r[i, j] = 2 * np.dot(fac, s[i, :] * np.cos(w * t[j]))
            else:
                r[i, j] = 2 * np.dot(fac, np.sqrt((s[i, :] * s[j, :])) * np.cos(w * (t[i] - t[j])))
    return r


def r_to_s(r, w, t):
    dt = t[1] - t[0]
    fac = np.ones(len(t))
    fac[1: len(t) - 1: 2] = 4
    fac[2: len(t) - 2: 2] = 2
    fac = fac * dt / 3

    s = np.zeros([r.shape[0], len(w)])
    for i in range(r.shape[0]):
        for j in range(len(w)):
            r[i, j] = 2 / (2 * np.pi) * np.dot(fac, (r[i, :] * np.cos(t * w[j])))
    s[s < 0] = 0
    return s
