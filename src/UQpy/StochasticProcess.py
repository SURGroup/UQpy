"""This module contains functionality for all the stochastic process generation supported by UQpy."""

from UQpy.tools import *
from UQpy.Distributions import *
from scipy.linalg import sqrtm
from scipy.stats import norm
import itertools


class SRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum density based on the Spectral Representation
    Method. This class can simulate both uni-variate and multi-variate multi-dimensional Stochastic Processes.

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param S: Power spectrum to be used for generating the samples
    :type S: numpy.ndarray

    :param dw: Array of frequency discretizations across dimensions
    :type dw: array

    :param nt: Array of number of time discretizations across dimensions
    :type nt: array

    :param nw: Array of number of frequency discretizations across dimensions
    :type nw: array

    :param case: Uni-variate or Multivariate options.
                    1. 'uni' - Uni-variate
                    2. 'multi' - Multi-variate

    :param g: The cross - Power Spectral Density. Used only for the Multi-variate case.
                    Default: None

    Output:
    :rtype: samples: numpy.ndarray
    """
    # Created by Lohit Vandanapu
    # Last Modified:08/04/2018 Lohit Vandanapu

    def __init__(self, nsamples, S, dw, nt, nw, case='uni', g=None):
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.nsamples = nsamples
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(size=np.append(self.nsamples, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.m = self.S.shape[0]
            self.n = len(S.shape[1:])
            self.g = g
            self.phi = np.random.uniform(size=np.append(self.nsamples, np.append(self.m, np.ones(self.n, dtype=np.int32) * self.nw))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        B = np.exp(phi * 1.0j) * np.sqrt(2 ** (self.n + 1) * self.S * np.prod(self.dw))
        sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
        samples = np.real(sample)
        return samples

    def _simulate_multi(self, phi):
        # Assembly of S_jk
        S_sqrt = np.sqrt(self.S)
        S_jk = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)
        # Assembly of g_jk
        g_jk = np.zeros_like(S_jk)
        l = 0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                g_jk[i, j] = self.g[l]
                l = l + 1
        g_jk = np.einsum('ij...->ji...', g_jk) + g_jk

        for i in range(self.m):
            g_jk[i, i] = np.ones_like(S_jk[0, 0])
        S = S_jk * g_jk

        S = np.einsum('ij...->...ij', S)
        S1 = S[..., :, :]
        H_jk = np.zeros_like(S1)
        for i in range(len(S1)):
            try:
                H_jk[i] = np.linalg.cholesky(S1[i])
            except:
                H_jk[i] = np.linalg.cholesky(nearestPD(S1[i]))
        H_jk = H_jk.reshape(S.shape)
        H_jk = np.einsum('...ij->ij...', H_jk)
        samples_list = []
        for i in range(self.m):
            samples = 0
            for j in range(i+1):
                B = H_jk[i, j] * np.sqrt(2 ** (self.n + 1) * np.prod(self.dw)) * np.exp(phi[:, j] * 1.0j)
                sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
                samples += np.real(sample)
            samples_list.append(samples)
        samples_list = np.array(samples_list)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return np.einsum('ij...->ji...', samples_list)


class BSRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum and bispectrum density based on the
    BiSpectral Representation Method.

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param S: Power Spectral Density to be used for generating the samples
    :type S: numpy.ndarray

    :param B: BiSpectral Density to be used for generating the samples
    :type B: numpy.ndarray

    :param dt: Array of time discretizations across dimensions
    :type dt: array

    :param dw: Array of frequency discretizations across dimensions
    :type dw: array

    :param nt: Array of number of time discretizations across dimensions
    :type nt: array

    :param nw: Array of number of frequency discretizations across dimensions
    :type nw: array

    Output:
    :rtype: samples: numpy.ndarray
    """
    # Created by Lohit Vandanapu
    # Last Modified:08/04/2018 Lohit Vandanapu

    def __init__(self, n_sim, S, B, dt, dw, nt, nw, case='uni', g=None):
        self.n_sim = n_sim
        self.nw = nw
        self.nt = nt
        self.dw = dw
        self.dt = dt
        self.n = len(S.shape)
        self.S = S
        self.B = B
        self.B_Ampl = np.absolute(B)
        self.B_Real = np.real(B)
        self.B_Imag = np.imag(B)
        self.Biphase = np.arctan2(self.B_Imag, self.B_Real)
        self.Biphase[np.isnan(self.Biphase)] = 0
        self.phi = np.random.uniform(size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
        self._compute_bicoherence()
        self.samples = self._simulate_bsrm_uni()

    def _compute_bicoherence(self):
        self.Bc2 = np.zeros_like(self.B_Real)
        self.PP = np.zeros_like(self.S)
        self.sum_Bc2 = np.zeros_like(self.S)

        if self.n == 1:
            self.PP[0] = self.S[0]
            self.PP[1] = self.S[1]

        if self.n == 2:
            self.PP[0, :] = self.S[0, :]
            self.PP[1, :] = self.S[1, :]
            self.PP[:, 0] = self.S[:, 0]
            self.PP[:, 1] = self.S[:, 1]

        if self.n == 3:
            self.PP[0, :, :] = self.S[0, :, :]
            self.PP[1, :, :] = self.S[1, :, :]
            self.PP[:, 0, :] = self.S[:, 0, :]
            self.PP[:, 1, :] = self.S[:, 1, :]
            self.PP[:, :, 0] = self.S[:, :, 0]
            self.PP[:, 0, 1] = self.S[:, :, 1]

        self.ranges = [range(self.nw) for _ in range(self.n)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                if self.B_Ampl[(*wi, *wj)] > 0 and self.PP[(*wi, *[])] * self.PP[(*wj, *[])] != 0:
                    self.Bc2[(*wi, *wj)] = self.B_Ampl[(*wi, *wj)] ** 2 / (
                                self.PP[(*wi, *[])] * self.PP[(*wj, *[])] * self.S[(*wk, *[])]) * self.dw ** self.n
                    self.sum_Bc2[(*wk, *[])] = self.sum_Bc2[(*wk, *[])] + self.Bc2[(*wi, *wj)]
                else:
                    self.Bc2[(*wi, *wj)] = 0
            if self.sum_Bc2[(*wk, *[])] > 1:
                print('Results may not be as expected as sum of partial bicoherences is greater than 1')
                for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.Bc2[(*wi, *wj)] = self.Bc2[(*wi, *wj)] / self.sum_Bc2[(*wk, *[])]
                self.sum_Bc2[(*wk, *[])] = 1
            self.PP[(*wk, *[])] = self.S[(*wk, *[])] * (1 - self.sum_Bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self):
        Coeff = np.sqrt((2 ** (self.n + 1)) * self.S * self.dw ** self.n)
        Phi_e = np.exp(self.phi * 1.0j)
        Biphase_e = np.exp(self.Biphase * 1.0j)
        B = np.sqrt(1 - self.sum_Bc2) * Phi_e
        Bc = np.sqrt(self.Bc2)

        Phi_e = np.einsum('i...->...i', Phi_e)
        B = np.einsum('i...->...i', B)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
                                Phi_e[(*wj, *[])]

        B = np.einsum('...i->i...', B)
        Phi_e = np.einsum('...i->i...', Phi_e)
        B = B * Coeff
        B[np.isnan(B)] = 0
        samples = np.fft.fftn(B, [self.nt for _ in range(self.n)])
        return np.real(samples)

class KLE:
    """
    A class to simulate Stochastic Processes from a given auto-correlation function based on the Karhunen-Louve Expansion

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param R: Auto-correlation Function to be used for generating the samples
    :type R: numpy.ndarray

    Output:
    :rtype: samples: numpy.ndarray
    """
    # Created by Lohit Vandanapu
    # Last Modified:08/04/2018 Lohit Vandanapu

    def __init__(self, nsamples, R):
        self.R = R
        self.samples = self._simulate(nsamples)

    def _simulate(self, nsamples):
        lam, phi = np.linalg.eig(self.R)
        nRV = self.R.shape[0]
        xi = np.random.normal(size=(nRV, nsamples))
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi, np.dot(sqrtm(lam), xi))
        samples = np.real(samples)
        samples = samples.T
        return samples


class Translate:
    """
    A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

    :param samples_g: Gaussian Stochastic Processes
    :type samples_g: numpy.ndarray

    :param R_g: Auto-correlation Function of the Gaussian Stochastic Processes
    :type R_g: numpy.ndarray

    :param marginal: list of marginal
    :type marginal: list

    :param params: list of parameters for the marginal
    :type params: list

    Output:
    :rtype: samples_ng: numpy.ndarray
    :rtype: R_ng: numpy.ndarray
    """
    # Created by Lohit Vandanapu
    # Last Modified:08/06/2018 Lohit Vandanapu

    def __init__(self, samples_g, R_g, marginal, params):
        self.samples_g = samples_g
        self.R_g = R_g
        self.num = self.R_g.shape[0]
        self.dim = len(self.R_g.shape)
        self.marginal = marginal
        self.params = params
        self.samples_ng = self.translate_g_samples()
        self.R_ng = self.autocorrealtion_distortion()

    def translate_g_samples(self):
        std = np.sqrt(np.diag(self.R_g)[0])
        samples_cdf = norm.cdf(self.samples_g, scale=std)
        samples_ng = inv_cdf(self.marginal)[0](samples_cdf, self.params[0])
        return samples_ng

    def autocorrealtion_distortion(self):
        r_g = R_to_r(self.R_g)
        r_g = np.clip(r_g, -0.999, 0.999)
        R_ng = np.zeros_like(r_g)
        for i in itertools.product(*[self.num for _ in range(self.dim)]):
            R_ng[(*i, *[])] = self.solve_integral(r_g[(*i, *[])])
        return R_ng

    def solve_integral(self, rho):
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
        tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        coef = tmp_f_xi * tmp_f_eta * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        return rho_non


class Inverse_Translate:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes

    :param samples_ng: Gaussian Stochastic Processes
    :type samples_ng: numpy.ndarray

    :param R_ng: Auto-correlation Function of the Gaussian Stochastic Processes
    :type R_ng: numpy.ndarray

    :param marginal: list of marginal
    :type marginal: list

    :param params: list of parameters for the marginal
    :type params: list

    Output:
    :rtype: samples_g: numpy.ndarray
    :rtype: R_g: numpy.ndarray
    """
    # Created by Lohit Vandanapu
    # Last Modified:08/06/2018 Lohit Vandanapu

    def __init__(self, samples_ng, R_ng, marginal, params):
        self.samples_ng = samples_ng
        self.R_ng = R_ng
        self.num = self.R_ng.shape[0]
        self.dim = len(self.R_ng.shape)
        self.marginal = marginal
        self.params = params
        self.samples_g = self.inverse_translate_ng_samples()
        self.R_g = self.itam()

    def inverse_translate_ng_samples(self):
        samples_cdf = cdf(self.marginal)[0](self.samples_ng, self.params[0])
        samples_g = inv_cdf(['Normal'])[0](samples_cdf, [[0, 1]])
        return samples_g

    def itam(self):
        # Initial Guess
        corr_norm0 = self.R_ng
        # Iteration Condition
        i_converge = 0
        error0 = 100
        max_iter = 20
        corr0 = np.zeros_like(corr_norm0)
        corr1 = np.zeros_like(corr_norm0)

        for ii in range(max_iter):
            for i in itertools.product(*[self.num for _ in range(self.dim)]):
                corr0[(*i, *[])] = self.solve_integral(corr_norm0[(*i, *[])])
            # compute the relative difference between the computed NGACF & the target R(Normalized)
            err1 = np.sum((self.R_ng - corr0) ** 2)
            err2 = np.sum(corr0 ** 2)
            error1 = 100 * np.sqrt(err1 / err2)

            if abs(error0 - error1) / error1 < 0.001 or ii == max_iter or 100 * np.sqrt(err1 / err2) < 0.0005:
                i_converge = 1

            corr_norm1 = (self.R_ng / corr0) * corr_norm0

            # Eliminate Numerical error of Upgrading Scheme
            corr_norm1[corr_norm1 < -1.0] = -0.99999
            corr_norm1[corr_norm1 > 1.0] = 0.99999

            # Iteratively finding the nearest PSD(Qi & Sun, 2006)
            corr_norm1 = np.array(near_pd(corr_norm1))

            if i_converge == 0 and ii != max_iter:
                corr_norm0 = corr_norm1
                error0 = error1

        return corr_norm1

    def solve_integral(self, rho):
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
        tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        coef = tmp_f_xi * tmp_f_eta * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        return rho_non


# def solve_integral_simpsons_rule(rho):
#     n = 512
#     zmax = 6
#     zmin = -zmax
#     x1 = np.linspace(zmin, zmax, n)
#     x2 = np.linspace(zmin, zmax, n)
#
#     tmp_f_x1 = inv_cdf(marginal)[0](stats.norm.cdf(x1), params[0])
#     tmp_f_x2 = inv_cdf(marginal)[0](stats.norm.cdf(x2), params[0])
#     phi = bi_variate_normal_pdf(x1[:, None], x2, rho)
#     integrand = tmp_f_x1[:, None]*tmp_f_x2*phi
#     rho_non = simps(simps(integrand, x2), x1)
#     return rho_non


# rho_n1 = np.zeros_like(rho1)
# op = solve_integral_simpsons_rule
# it = np.nditer([rho1, rho_n1], op_flags=[['readonly'], ['writeonly']])
# for i in range(401):
#     for j in range(i, 401):
#         rho_n1[i, j] = op(rho1[i, j])

