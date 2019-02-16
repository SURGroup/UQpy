"""This module contains functionality for all the stochastic process generation supported by UQpy."""

from UQpy.Utilities import *
from UQpy.Distributions import *
from scipy.linalg import sqrtm
from scipy.stats import norm
import itertools


class SRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum density based on the Spectral Representation
    Method. This class can simulate both uni-variate and multi-variate multi-dimensional Stochastic Processes. Uses
    Singular Value Decomposition as opposed to Cholesky Decomposition to be more robust with near-Positive Definite
    multi-dimensional Power Spectra.

    Input:

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param S: Power spectrum to be used for generating the samples
    :type S: numpy.ndarray

    :param dw: List of frequency discretizations across dimensions
    :type dw: list

    :param nt: List of number of time discretizations across dimensions
    :type nt: list

    :param nw: List of number of frequency discretizations across dimensions
    :type nw: list

    :param case: Uni-variate or Multivariate options.
                    1. 'uni' - Uni-variate
                    2. 'multi' - Multi-variate
    :type case: str

    Output:

    :rtype: samples: numpy.ndarray
    """

    # Created by Lohit Vandanapu
    # Last Modified:02/12/2019 Lohit Vandanapu

    def __init__(self, nsamples, S, dw, nt, nw, case='uni'):
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.nsamples = nsamples
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(
                size=np.append(self.nsamples, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.m = self.S.shape[0]
            self.n = len(S.shape[2:])
            self.phi = np.random.uniform(
                size=np.append(self.nsamples, np.append(np.ones(self.n, dtype=np.int32) * self.nw, self.m))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        B = np.exp(phi * 1.0j) * np.sqrt(2 ** (self.n + 1) * self.S * np.prod(self.dw))
        sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
        samples = np.real(sample)
        return samples

    def _simulate_multi(self, phi):
        S = np.einsum('ij...->...ij', self.S)
        Coeff = np.sqrt(2 ** (self.n + 1)) * np.sqrt(np.prod(self.dw))
        U, s, V = np.linalg.svd(S)
        R = np.einsum('...ij,...j->...ij', U, np.sqrt(s))
        F = Coeff * np.einsum('...ij,n...j -> n...i', R, np.exp(phi * 1.0j))
        F[np.isnan(F)] = 0
        samples = np.real(np.fft.fftn(F, s=[self.nt for _ in range(self.n)], axes=tuple(np.arange(1, 1+self.n))))
        return samples

class BSRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum and bispectrum density based on the BiSpectral
    Representation Method.This class can simulate both uni-variate and multi-variate multi-dimensional Stochastic
    Processes. This class uses Singular Value Decomposition as opposed to Cholesky Decomposition to be more robust with
    near-Positive Definite multi-dimensional Power Spectra.

    Input:

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param S: Power Spectral Density to be used for generating the samples
    :type S: numpy.ndarray

    :param B: BiSpectral Density to be used for generating the samples
    :type B: numpy.ndarray

    :param dt: Array of time discretizations across dimensions
    :type dt: numpy.ndarray

    :param dw: Array of frequency discretizations across dimensions
    :type dw: numpy.ndarray

    :param nt: Array of number of time discretizations across dimensions
    :type nt: numpy.ndarray

    :param nw: Array of number of frequency discretizations across dimensions
    :type nw: numpy.ndarray

    Output:

    :rtype samples: numpy.ndarray
    """

    # Created by Lohit Vandanapu
    # Last Modified:02/12/2019 Lohit Vandanapu

    def __init__(self, n_sim, S, B, dt, dw, nt, nw, case='uni'):
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
    A class to simulate Stochastic Processes from a given auto-correlation function based on the Karhunen-Louve
    Expansion

    Input:

    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param R: Auto-correlation Function to be used for generating the samples
    :type R: numpy.ndarray

    Output:

    :rtype samples: numpy.ndarray
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


class Translation:
    """
    A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

    Input:

    :param samples_g: Gaussian Stochastic Processes
    :type samples_g: numpy.ndarray

    :param S_g: Power Spectrum of the Gaussian Stochastic Processes
    :type S_g: numpy.ndarray

    :param R_g: Auto-correlation Function of the Gaussian Stochastic Processes
    :type R_g: numpy.ndarray

    :param marginal: name of marginal
    :type marginal: str

    :param params: list of parameters for the marginal
    :type params: list

    Output:

    :rtype samples_ng: numpy.ndarray
    :rtype R_ng: numpy.ndarray
    """

    # Created by Lohit Vandanapu
    # Last Modified:02/12/2019 Lohit Vandanapu

    def __init__(self, samples_g, marginal, params, dt, dw, nt, nw, S_g=None, R_g=None):
        self.samples_g = samples_g
        if R_g or S_g is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        else:
            if R_g is None:
                self.S_g = S_g
                self.R_g = r_to_s(S_g, np.arange(0, nw)*dw, np.arange(0, nt)*dt)
            elif S_g is None:
                self.R_g = R_g
                self.S_g = s_to_r(R_g, np.arange(0, nw)*dw, np.arange(0, nt)*dt)
        self.num = self.R_g.shape[0]
        self.dim = len(self.R_g.shape)
        self.marginal = marginal
        self.params = params
        self.samples_ng = self.translate_g_samples()
        self.R_ng = self.autocorrealtion_distortion()
        self.S_ng = r_to_s(self.R_ng, np.arange(0, nw)*dw, np.arange(0, nt)*dt)

    def translate_g_samples(self):
        std = np.sqrt(np.var(self.samples_g))
        samples_cdf = norm.cdf(self.samples_g, scale=std)
        # samples_ng = inv_cdf(self.marginal)[0](samples_cdf, self.params[0])
        samples_ng = Distribution(self.marginal, self.params).icdf(samples_cdf, self.params)
        return samples_ng

    def autocorrealtion_distortion(self):
        # r_g = R_to_r(self.R_g)
        r_g = np.clip(r_g, -0.999, 0.999)
        R_ng = np.zeros_like(r_g)
        for i in itertools.product(*[range(self.num) for _ in range(self.dim)]):
            R_ng[(*i, *[])] = self.solve_integral(r_g[(*i, *[])])
        return R_ng

    def solve_integral(self, rho):
        if rho == 1.0:
            rho = 0.999
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
        # tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        # tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        tmp_f_xi = Distribution(self.marginal, self.params).icdf(stats.norm.cdf(xi), self.params)
        tmp_f_eta = Distribution(self.marginal, self.params).icdf(stats.norm.cdf(eta), self.params)
        coef = tmp_f_xi * tmp_f_eta * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        rho_non = (rho_non - (Distribution(self.marginal, self.params).moments(self.params)[0]) ** 2) / \
                  Distribution(self.marginal, self.params).moments(self.params)[1]
        return rho_non


class InverseTranslation:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes

    Input:

    :param samples_ng: Gaussian Stochastic Processes
    :type samples_ng: numpy.ndarray

    :param R_ng: Auto-correlation Function of the Gaussian Stochastic Processes
    :type R_ng: numpy.ndarray

    :param marginal: mane of the marginal
    :type marginal: str

    :param params: list of parameters for the marginal
    :type params: list

    Output:

    :rtype samples_g: numpy.ndarray
    :rtype R_g: numpy.ndarray
    """

    # Created by Lohit Vandanapu
    # Last Modified:02/13/2019 Lohit Vandanapu

    def __init__(self, samples_ng, marginal, params, dt, dw, nt, nw, R_ng=None, S_ng=None):
        self.samples_ng = samples_ng
        self.w = np.arange(0, nw)*dw
        self.t = np.arange(0, nt)*dt
        if R_ng or S_ng is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        else:
            if R_ng is None:
                self.S_ng = S_ng
                self.R_ng = s_to_r(S_ng, self.w, self.t)
            elif S_ng is None:
                self.R_ng = R_ng
                self.S_ng = r_to_s(R_ng, self.w, self.t)
        self.num = self.R_ng.shape[0]
        self.dim = len(self.R_ng.shape)
        self.marginal = marginal
        self.params = params
        self.samples_g = self.inverse_translate_ng_samples()
        self.S_g = self.itam()
        self.R_r = s_to_r(self.S_g, self.w, self.t)

    def inverse_translate_ng_samples(self):
        # samples_cdf = cdf(self.marginal)[0](self.samples_ng, self.params[0])
        # samples_g = inv_cdf(['Normal'])[0](samples_cdf, [0, 1])
        samples_cdf = Distribution(self.marginal, self.params).cdf(self.samples_ng, self.params)
        samples_g = Distribution('Normal', [0, 1]).icdf(samples_cdf, [0, 1])
        return samples_g

    def itam(self):
        # Initial Guess
        target_s = self.S_ng
        # Iteration Conditions
        i_converge = 0
        error0 = 100
        max_iter = 1
        target_r = s_to_r(target_s, self.w, self.t)
        r_g_iterate = target_r
        s_g_iterate = target_s
        r_ng_iterate = np.zeros_like(target_r)
        s_ng_iterate = np.zeros_like(target_s)

        for ii in range(max_iter):
            # for i in itertools.product(*[range(self.num) for _ in range(self.dim)]):
            for i in range(len(target_r)):
                r_ng_iterate[i] = self.solve_integral(r_g_iterate)
            s_ng_iterate = s_to_r(r_ng_iterate, self.w, self.t)

            # compute the relative difference between the computed NGACF & the target R(Normalized)
            err1 = np.sum((target_s - s_ng_iterate) ** 2)
            err2 = np.sum(target_s ** 2)
            error1 = 100 * np.sqrt(err1 / err2)

            if abs(error0 - error1) / error1 < 0.001 or ii == max_iter or 100 * np.sqrt(err1 / err2) < 0.0005:
                i_converge = 1

            s_g_next_iterate = (target_s / s_ng_iterate) * s_g_iterate

            # Eliminate Numerical error of Upgrading Scheme
            s_g_next_iterate[s_g_next_iterate < 0] = 0

            if i_converge == 0 and ii != max_iter:
                s_g_iterate = s_g_next_iterate
                error0 = error1

        return s_g_iterate

    def solve_integral(self, rho):
        if rho == 1.0:
            rho = 0.999
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
        tmp_f_xi = Distribution(self.marginal, self.params).icdf(stats.norm.cdf(xi), self.params)
        tmp_f_eta = Distribution(self.marginal, self.params).icdf(stats.norm.cdf(eta), self.params)
        # tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        # tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        coef = tmp_f_xi * tmp_f_eta * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        rho_non = (rho_non - (Distribution(self.marginal, self.params).moments(self.params)[0]) ** 2) / \
                  Distribution(self.marginal, self.params).moments(self.params)[1]
        return rho_non
