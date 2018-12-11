import numpy as np
import scipy.stats as stats
from scipy import interpolate
from scipy.integrate import simps


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q * xdiag * Q.T


def _getPs(A, W=None):
    W05 = np.matrix(W ** .5)
    return W05.I * _getAplus(W05 * A * W05) * W05.I


def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def near_pd(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def estimate_PSD(samples, nt, T):
    sample_size = nt
    sample_maxtime = T
    dt = T / (nt - 1)
    Xw = np.fft.fft(samples, sample_size, axis=1)
    Xw = Xw[:, 0: int(sample_size / 2)]
    m_Ps = np.mean(np.absolute(Xw) ** 2 * sample_maxtime / sample_size ** 2, axis=0)
    num = int(T / (2 * dt))
    return np.linspace(0, (1 / (2 * dt) - 1 / T), num), m_Ps


def S_to_R(S, w, t):
    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    R = np.zeros([S.shape[0], len(t)])
    for i in range(S.shape[0]):
        for j in range(len(t)):
            if S.shape[0] == 1:
                # np.array(2 * np.multiply(np.cos(np.matmul(np.transpose(np.matrix(t)), np.matrix(w))), S_NGT[i, :])*np.transpose(np.matrix(fac))).flatten()
                R[i, j] = 2 * np.dot(fac, S[i, :] * np.cos(w * t[j]))
            else:
                R[i, j] = 2 * np.dot(fac, np.sqrt((S[i, :] * S[j, :])) * np.cos(w * (t[i] - t[j])))
    return R


def R_to_r(R):
    # Normalize target non - Gaussian Correlation function to Correlation coefficient
    r = np.zeros_like(R)
    dim = len(R.shape)
    if dim == 1:
        r = R/R[0]
    else:
        for i in range(R.shape[0]):
            # Stationary
            index = [i]*dim
            if R[(*index, *[])] != 0:
                r[i] = R[i] / R[(*index, *[])]
            else:
                r[i] = 0
    return r


def dist_to_R(r, params):
    # Code as of now works for 2 dimensions
    # Normalize target non - Gaussian Correlation function to Correlation coefficient
    R = np.zeros_like(r)
    for i in range(R.shape[0]):
        # Stationary
        if R[i, i] != 0:
            R[i, :] = r[i, :]
        else:
            R[i, :] = 0
    return R


def R_to_S(R, w, t):
    dt = t[1] - t[0]
    fac = np.ones(len(t))
    fac[1: len(t) - 1: 2] = 4
    fac[2: len(t) - 2: 2] = 2
    fac = fac * dt / 3

    S = np.zeros([R.shape[0], len(w)])
    for i in range(R.shape[0]):
        for j in range(len(w)):
            S[i, j] = 2 / (2 * np.pi) * np.dot(fac, (R[i, :] * np.cos(t * w[j])))
    S[S < 0] = 0
    return S


def translate(R_G, name, pseudo, mu, sig, parameter1, parameter2):
    R_NG = np.zeros_like(R_G)

    if name == 'Lognormal_Distribution':
        for i in range(R_G.shape[0]):
            sigmaN1 = parameter1[i]
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 0])
            else:
                sy1 = np.sqrt(R_G[i, i])
            muN1 = 0.5 * np.log(sig[i] ** 2 / (np.exp(sigmaN1 ** 2) - 1)) - 0.5 * sigmaN1 ** 2
            shift1 = -np.exp(muN1 + 0.5 * sigmaN1 ** 2)
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 0:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1, shift2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                            #                             shift1,
                            #                             shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
                else:
                    sigmaN2 = parameter1[j]
                    sy2 = np.sqrt(R_G[j, j])
                    muN2 = 0.5 * np.log(sig[j] ** 2 / (np.exp(sigmaN2 ** 2) - 1)) - 0.5 * sigmaN2 ** 2
                    shift2 = -np.exp(muN2 + 0.5 * sigmaN2 ** 2)
                    if sy1 != 0 and sy2 != 0:
                        if i != j:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1, shift2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                            #                             shift1,
                            #                             shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
    return R_NG


def translate_process(Samples_G, Dist, mu, sig, parameter1, parameter2):
    Samples_NG = np.zeros_like(Samples_G)
    if Dist == 'Lognormal':
        for i in range(len(Samples_G)):
            sy1 = 1
            sigmaN1 = parameter1[i]
            muN1 = 0.5 * np.log(sig[i] ** 2 / (np.exp(sigmaN1 ** 2) - 1)) - 0.5 * sigmaN1 ** 2
            shift1 = -np.exp(muN1 + 0.5 * sigmaN1 ** 2)
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = stats.lognorm.ppf(fg1, muN1, sigmaN1)
            g1 = g1 + shift1
            Samples_NG[i, :] = mu[i] + g1
    elif Dist == 'Beta':
        for i in range(len(Samples_G)):
            sy1 = 1
            alpha = parameter1[i]
            beta = parameter2[i]
            lo_lim1 = 0. - sig[i] * np.sqrt(alpha * (alpha + beta + 1) / beta)
            hi_lim1 = 0. + sig[i] * np.sqrt(beta * (alpha + beta + 1) / alpha)
            stretch1 = hi_lim1 - lo_lim1
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = stats.beta.ppf(fg1, alpha, beta)
            g1 = g1 * stretch1 + lo_lim1
            Samples_NG[i, :] = mu[i] + g1
    elif Dist == 'User':
        for i in range(len(Samples_G)):
            sy1 = 1
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = interpolate.interp1d(parameter2, parameter1)
            g1 = g1(fg1)
            Samples_NG[i, :] = g1
    return Samples_NG


def solve_double_integral(marginal, params, rho_norm):
    """
        A function to solve the double integral equation in order to evaluate the modified correlation matrix in the
        standard normal space given the correlation matrix in the original space. This is achieved by a quadratic
        two-dimensional Gauss-Legendre integration.
    """

    n = 2
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
            tmp_f_xi = ((icdf_j(stats.norm.cdf(xi), params[j]) - mj[0]) / np.sqrt(mj[1]))
            tmp_f_eta = ((icdf_i(stats.norm.cdf(eta), params[i]) - mi[0]) / np.sqrt(mi[1]))
            coef = tmp_f_xi * tmp_f_eta * w2d

            rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
            rho[j, i] = rho[i, j]

    return rho


def bi_variate_normal_pdf(x1, x2, rho):
    """
        A function which evaluates the values of the bi-variate normal probability distribution function
    """
    return (1 / (2 * np.pi * np.sqrt(1-rho**2)) *
            np.exp(-1/(2*(1-rho**2)) *
                   (x1**2 - 2 * rho * x1 * x2 + x2**2)))


def itam(marginal, params, corr):
    # Initial Guess
    corr_norm0 = corr
    # Iteration Condition
    i_converge = 0
    error0 = 100
    max_iter = 20

    for ii in range(max_iter):
        corr0 = solve_double_integral(marginal, params, corr_norm0)
        # compute the relative difference between the computed NGACF & the target R(Normalized)
        err1 = 1.0e-10
        err2 = 1.0e-10
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