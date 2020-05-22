# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.special import gamma
from scipy.stats import chi2, norm


def run_parallel_python(model_script, model_object_name, sample, dict_kwargs=None):
    """
    Execute the python model in parallel
    :param sample: One sample point where the model has to be evaluated
    :return:
    """

    exec('from ' + model_script[:-3] + ' import ' + model_object_name)
    # if kwargs is not None:
    #     par_res = eval(model_object_name + '(sample, kwargs)')
    # else:
    if dict_kwargs is None:
        par_res = eval(model_object_name + '(sample)')
    else:
        par_res = eval(model_object_name + '(sample, **dict_kwargs)')
    # par_res = parallel_output
    # if self.model_is_class:
    #     par_res = parallel_output.qoi
    # else:
    #     par_res = parallel_output

    return par_res


# def compute_Voronoi_volume(vertices):
#
#     from scipy.spatial import Delaunay
#
#     d = Delaunay(vertices)
#     d_vol = np.zeros(np.size(vertices, 0))
#     for i in range(d.nsimplex):
#         d_verts = vertices[d.simplices[i]]
#         d_vol[i] = compute_Delaunay_volume(d_verts)
#
#     volume = np.sum(d_vol)
#     return volume


def voronoi_unit_hypercube(samples):

    from scipy.spatial import Voronoi

    # Mirror the samples in both low and high directions for each dimension
    samples_center = samples
    dimension = samples.shape[1]
    for i in range(dimension):
        samples_del = np.delete(samples_center, i, 1)
        if i == 0:
            points_temp1 = np.hstack([np.atleast_2d(-samples_center[:,i]).T, samples_del])
            points_temp2 = np.hstack([np.atleast_2d(2-samples_center[:,i]).T, samples_del])
        elif i == dimension-1:
            points_temp1 = np.hstack([samples_del, np.atleast_2d(-samples_center[:, i]).T])
            points_temp2 = np.hstack([samples_del, np.atleast_2d(2 - samples_center[:, i]).T])
        else:
            points_temp1 = np.hstack([samples_del[:,:i], np.atleast_2d(-samples_center[:, i]).T, samples_del[:,i:]])
            points_temp2 = np.hstack([samples_del[:,:i], np.atleast_2d(2 - samples_center[:, i]).T, samples_del[:,i:]])
        samples = np.append(samples, points_temp1, axis=0)
        samples = np.append(samples, points_temp2, axis=0)

    vor = Voronoi(samples, incremental=True)

    eps = sys.float_info.epsilon
    regions = [None]*samples_center.shape[0]

    for i in range(samples_center.shape[0]):
        regions[i] = vor.regions[vor.point_region[i]]

    # for region in vor.regions:
    #     flag = True
    #     for index in region:
    #         if index == -1:
    #             flag = False
    #             break
    #         else:
    #             for i in range(dimension):
    #                 x = vor.vertices[index, i]
    #                 if not (-eps <= x and x <= 1 + eps):
    #                     flag = False
    #                     break
    #     if region != [] and flag:
    #         regions.append(region)

    vor.bounded_points = samples_center
    vor.bounded_regions = regions

    return vor


def compute_Voronoi_centroid_volume(vertices):

    from scipy.spatial import Delaunay, ConvexHull

    T = Delaunay(vertices)
    dimension = np.shape(vertices)[1]

    w = np.zeros((T.nsimplex, 1))
    cent = np.zeros((T.nsimplex, dimension))
    for i in range(T.nsimplex):
        ch = ConvexHull(T.points[T.simplices[i]])
        w[i] = ch.volume
        cent[i, :] = np.mean(T.points[T.simplices[i]], axis=0)
    V = np.sum(w)
    C = np.matmul(np.divide(w, V).T, cent)

    return C, V


def compute_Delaunay_centroid_volume(vertices):

    from scipy.spatial import ConvexHull

    ch = ConvexHull(vertices)
    volume = ch.volume
    centroid = np.mean(vertices, axis=0)

    # v1 = np.concatenate((np.ones([np.size(vertices, 0), 1]), vertices), 1)
    # volume = (1 / math.factorial(np.size(vertices, 0) - 1)) * np.linalg.det(v1.T)

    return centroid, volume


def correlation_distortion(marginal, rho_norm):

    """
        Description:

            A function to solve the double integral equation in order to evaluate the modified correlation
            matrix in the standard normal space given the correlation matrix in the original space. This is achieved
            by a quadratic two-dimensional Gauss-Legendre integration.
            This function is a part of the ERADIST code that can be found in:
            https://www.era.bgu.tum.de/en/software/

        Input:
            :param marginal: marginal distributions
            :type marginal: list

            :param params: marginal distribution parameters.
            :type params: list

            :param rho_norm: Correlation at standard normal space.
            :type rho_norm: ndarray

        Output:
            :return rho: Distorted correlation
            :rtype rho: ndarray

    """

    n = 1024
    z_max = 8
    z_min = -z_max
    points, weights = np.polynomial.legendre.leggauss(n)
    points = - (0.5 * (points + 1) * (z_max - z_min) + z_min)
    weights = weights * (0.5 * (z_max - z_min))

    xi = np.tile(points, [n, 1])
    xi = xi.flatten(order='F')
    eta = np.tile(points, n)

    first = np.tile(weights, n)
    first = np.reshape(first, [n, n])
    second = np.transpose(first)

    weights2d = first * second
    w2d = weights2d.flatten()
    rho = np.ones_like(rho_norm)

    print('UQpy: Computing Nataf correlation distortion...')
    from UQpy.Distributions import JointInd
    if isinstance(marginal, JointInd):
        if all(hasattr(m, 'moments') for m in marginal.marginals) and \
                all(hasattr(m, 'icdf') for m in marginal.marginals):
            for i in range(len(marginal.marginals)):
                i_cdf_i = marginal.marginals[i].icdf
                mi = marginal.marginals[i].moments()
                if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
                    raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")
                for j in range(i + 1, len(marginal.marginals)):
                    i_cdf_j = marginal.marginals[j].icdf
                    mj = marginal.marginals[j].moments()
                    if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                        raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

                    tmp_f_xi = ((i_cdf_j(np.atleast_2d(stats.norm.cdf(xi)).T) - mj[0]) / np.sqrt(mj[1]))
                    tmp_f_eta = ((i_cdf_i(np.atleast_2d(stats.norm.cdf(eta)).T) - mi[0]) / np.sqrt(mi[1]))
                    coef = tmp_f_xi * tmp_f_eta * w2d

                    rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
                    rho[j, i] = rho[i, j]

    elif isinstance(marginal, list):
        if all(hasattr(m, 'moments') for m in marginal) and \
                all(hasattr(m, 'icdf') for m in marginal):
            for i in range(len(marginal)):
                i_cdf_i = marginal[i].icdf
                mi = marginal[i].moments()
                if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
                    raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

                for j in range(i + 1, len(marginal)):
                    i_cdf_j = marginal[j].icdf
                    mj = marginal[j].moments()
                    if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                        raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

                    tmp_f_xi = ((i_cdf_j(np.atleast_2d(stats.norm.cdf(xi)).T) - mj[0]) / np.sqrt(mj[1]))
                    tmp_f_eta = ((i_cdf_i(np.atleast_2d(stats.norm.cdf(eta)).T) - mi[0]) / np.sqrt(mi[1]))
                    coef = tmp_f_xi * tmp_f_eta * w2d

                    rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
                    rho[j, i] = rho[i, j]

    print('UQpy: Done.')
    return rho


def itam(marginal, corr, beta, thresh1, thresh2):

    """
        Description:

            A function to perform the  Iterative Translation Approximation Method;  an iterative scheme for
            upgrading the Gaussian power spectral density function.
            [1] Shields M, Deodatis G, Bocchini P. A simple and efficient methodology to approximate a general
            non-Gaussian  stochastic process by a translation process. Probab Eng Mech 2011;26:511–9.


        Input:
            :param marginal: marginal distributions
            :type marginal: list

            :param params: marginal distribution parameters.
            :type params: list

            :param corr: Non-Gaussian Correlation matrix.
            :type corr: ndarray

            :param beta:  A variable selected to optimize convergence speed and desired accuracy.
            :type beta: float

            :param thresh1: Threshold
            :type thresh1: float

            :param thresh2: Threshold
            :type thresh2: float

        Output:
            :return corr_norm: Gaussian correlation matrix
            :rtype corr_norm: ndarray

    """

    if beta is None:
        beta = 1.0
    if thresh1 is None:
        thresh1 = 0.0001
    if thresh2 is None:
        thresh2 = 0.01

    # Initial Guess
    corr_norm0 = corr
    corr_norm = np.zeros_like(corr_norm0)
    # Iteration Condition
    error0 = 0.1
    error1 = 100.
    max_iter = 50
    iter_ = 0

    print("UQpy: Initializing Iterative Translation Approximation Method (ITAM)")
    while iter_ < max_iter and error1 > thresh1 and abs(error1-error0)/error0 > thresh2:
        error0 = error1
        corr0 = correlation_distortion(marginal, corr_norm0)
        error1 = np.linalg.norm(corr - corr0)

        max_ratio = np.amax(np.ones((len(corr), len(corr))) / abs(corr_norm0))

        corr_norm = np.nan_to_num((corr / corr0)**beta * corr_norm0)

        # Do not allow off-diagonal correlations to equal or exceed one
        corr_norm[corr_norm < -1.0] = (max_ratio + 1) / 2 * corr_norm0[corr_norm < -1.0]
        corr_norm[corr_norm > 1.0] = (max_ratio + 1) / 2 * corr_norm0[corr_norm > 1.0]

        # Iteratively finding the nearest PSD(Qi & Sun, 2006)
        corr_norm = np.array(nearest_psd(corr_norm))

        corr_norm0 = corr_norm.copy()

        iter_ = iter_ + 1

        print(["UQpy: ITAM iteration number ", iter_])
        print(["UQpy: Current error, ", error1])

    print("UQpy: ITAM Done.")
    return corr_norm


def bi_variate_normal_pdf(x1, x2, rho):

    """

        Description:

            A function which evaluates the values of the bi-variate normal probability distribution function.

        Input:
            :param x1: value 1
            :type x1: ndarray

            :param x2: value 2
            :type x2: ndarray

            :param rho: correlation between x1, x2
            :type rho: float

        Output:

    """
    return (1 / (2 * np.pi * np.sqrt(1-rho**2)) *
            np.exp(-1/(2*(1-rho**2)) *
                   (x1**2 - 2 * rho * x1 * x2 + x2**2)))


def _get_a_plus(a):

    """
        Description:

            A supporting function for the nearest_pd function

        Input:
            :param a:A general nd array

        Output:
            :return a_plus: A modified nd array
            :rtype:np.ndarray
    """

    eig_val, eig_vec = np.linalg.eig(a)
    q = np.matrix(eig_vec)
    x_diagonal = np.matrix(np.diag(np.maximum(eig_val, 0)))

    return q * x_diagonal * q.T


def _get_ps(a, w=None):

    """
        Description:

            A supporting function for the nearest_pd function

    """

    w05 = np.matrix(w ** .5)

    return w05.I * _get_a_plus(w05 * a * w05) * w05.I


def _get_pu(a, w=None):

    """
        Description:

            A supporting function for the nearest_pd function

    """

    a_ret = np.array(a.copy())
    a_ret[w > 0] = np.array(w)[w > 0]
    return np.matrix(a_ret)


def nearest_psd(a, nit=10):

    """
        Description:
            A function to compute the nearest positive semi definite matrix of a given matrix

         Input:
            :param a: Input matrix
            :type a: ndarray

            :param nit: Number of iterations to perform (Default=10)
            :type nit: int

        Output:
            :return:
    """

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

    """
        Description:

            Find the nearest positive-definite matrix to input
            A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
            credits [2].
            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Input:
            :param a: Input matrix
            :type a:


        Output:

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
    # `spacing` will, for Gaussian random matrices of small dimension, be on
    # other order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    k = 1
    while not is_pd(a3):
        min_eig = np.min(np.real(np.linalg.eigvals(a3)))
        a3 += np.eye(a.shape[0]) * (-min_eig * k**2 + spacing)
        k += 1

    return a3


def is_pd(b):

    """
        Description:

            Returns true when input is positive-definite, via Cholesky decomposition.

        Input:
            :param b: A general matrix

        Output:

    """
    try:
        _ = np.linalg.cholesky(b)
        return True
    except np.linalg.LinAlgError:
        return False


def estimate_psd(samples, nt, t):

    """
        Description: A function to estimate the Power Spectrum of a stochastic process given an ensemble of samples

        Input:
            :param samples: Samples of the stochastic process
            :param nt: Number of time discretisations in the time domain
            :param t: Total simulation time

        Output:
            :return: Power Spectrum
            :rtype: ndarray

    """

    sample_size = nt
    sample_max_time = t
    dt = t / (nt - 1)
    x_w = np.fft.fft(samples, sample_size, axis=1)
    x_w = x_w[:, 0: int(sample_size / 2)]
    m_ps = np.mean(np.absolute(x_w) ** 2 * sample_max_time / sample_size ** 2, axis=0)
    num = int(t / (2 * dt))

    return np.linspace(0, (1 / (2 * dt) - 1 / t), num), m_ps


def S_to_R(S, w, t):

    """
        Description:

            A function to transform the power spectrum to an autocorrelation function

        Input:
            :param s: Power Spectrum of the signal
            :param w: Array of frequency discretisations
            :param t: Array of time discretisations

        Output:
            :return r: Autocorrelation function
            :rtype: ndarray
    """

    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    R = np.zeros(len(t))
    for i in range(len(t)):
        R[i] = 2 * np.dot(fac, S * np.cos(w * t[i]))
    return R


def R_to_S(R, w, t):

    """
        Description: A function to transform the autocorrelation function to a power spectrum


        Input:
            :param r: Autocorrelation function of the signal
            :param w: Array of frequency discretizations
            :param t: Array of time discretizations

        Output:
            :return s: Power Spectrum
            :rtype: ndarray

    """
    dt = t[1] - t[0]
    fac = np.ones(len(t))
    fac[1: len(t) - 1: 2] = 4
    fac[2: len(t) - 2: 2] = 2
    fac = fac * dt / 3
    S = np.zeros(len(w))
    for i in range(len(w)):
        S[i] = 2 / (2 * np.pi) * np.dot(fac, R * np.cos(t * w[i]))
    S[S < 0] = 0
    return S


def R_to_r(R):

    """
        Description: A function to scale down the autocorrelation function to a correlation function


        Input:
            :param R: Autocorrelation function of the signal
        Output:
            :return r: correlation function of the signal
            :rtype: ndarray

    """
    r = R/np.max(R)
    return r


def IS_diagnostics(sampling_outputs=None, weights=None, graphics=False, figsize=(8, 3), ):
    """
    Diagnostics for IS.

    These diagnostics are qualitative, they can help the user in understanding how the IS algorithm is performing.
    This function returns printouts and plots.

    **Inputs:**

    :param sampling_outputs: output object of a sampling method
    :type sampling_outputs: object of class MCMC

    :param weights: output weights (alternative to giving sampling_outputs)
    :type weights: ndarray

    :param graphics: indicates whether or not to do a plot

                     Default: False
    :type graphics: boolean

    :param figsize: size of the figure for output plots
    :type figsize: tuple (width, height)

    """

    if (sampling_outputs is None) and (weights is None):
        raise ValueError('UQpy error: sampling_outputs or weights should be provided')
    if sampling_outputs is not None:
        weights = sampling_outputs.weights
    print('Diagnostics for Importance Sampling \n')
    effective_sample_size = 1/np.sum(weights**2, axis=0)
    print('Effective sample size is ne={}, out of a total number of samples={} \n'.
          format(effective_sample_size,np.size(weights)))
    print('max_weight = {}, min_weight = {} \n'.format(max(weights), min(weights)))

    # Output plots
    if graphics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(weights, np.zeros((np.size(weights), )), s=weights * 300, marker='o')
        ax.set_xlabel('weights')
        ax.set_title('Normalized weights out of importance sampling')
        plt.show(fig)


def MCMC_diagnostics(samples=None, sampling_outputs=None, eps_ESS=0.05, alpha_ESS=0.05,
                     graphics=False, figsize=None):
    """
    Diagnostics for MCMC.

    These diagnostics are qualitative, they can help the user in understanding how the MCMC algorithm is performing.
    These diagnostics are not intended to give a quantitative assessment of MCMC algorithms. This function returns
    printouts and plots.

    **Inputs:**

    :param sampling_outputs: output object of a sampling method
    :type sampling_outputs: object of class MCMC

    :param samples: output samples of a sampling method, alternative to giving sampling_outputs
    :type samples: ndarray

    :param eps_ESS: small number required to compute ESS when sampling_method='MCMC', see documentation
    :type eps_ESS: float in [0,1]

    :param alpha_ESS: small number required to compute ESS when sampling_method='MCMC', see documentation
    :type alpha_ESS: float in [0,1]

    :param graphics: indicates whether or not to do a plot

                     Default: False
    :type graphics: boolean

    :param figsize: size of the figure for output plots
    :type figsize: tuple (width, height)

    """

    if (eps_ESS < 0) or (eps_ESS > 1):
        raise ValueError('eps_ESS should be a float between 0 and 1.')
    if (alpha_ESS < 0) or (alpha_ESS > 1):
        raise ValueError('alpha_ESS should be a float between 0 and 1.')

    if (sampling_outputs is None) and (samples is None):
        raise ValueError('sampling_outputs or samples should be provided')
    if samples is None and sampling_outputs is not None:
        samples = sampling_outputs.samples

    if len(samples.shape) == 2:
        print('Diagnostics for a single chain of MCMC \n')
        print('!!! Warning !!! These diagnostics are purely qualitative and should be used with caution \n')
        nsamples, dim = samples.shape

        # Acceptance rate
        if sampling_outputs is not None:
            print('Acceptance ratio of the chain(s) = {}. \n'.format(sampling_outputs.acceptance_rate[0]))

        # Computation of ESS and min ESS
        bn = np.ceil(nsamples**(1/2))    # nb of samples per bin
        an = int(np.ceil(nsamples/bn))    # nb of bins
        idx = np.array_split(np.arange(nsamples), an)

        means_subdivisions = np.empty((an, samples.shape[1]))
        for i, idx_i in enumerate(idx):
            x_sub = samples[idx_i, :]
            means_subdivisions[i, :] = np.mean(x_sub, axis=0)
        Omega = np.cov(samples.T)
        Sigma = np.cov(means_subdivisions.T)
        joint_ESS = nsamples*np.linalg.det(Omega)**(1/dim)/np.linalg.det(Sigma)**(1/dim)
        chi2_value = chi2.ppf(1 - alpha_ESS, df=dim)
        min_joint_ESS = 2 ** (2 / dim) * np.pi / (dim * gamma(dim / 2)) ** (
                    2 / dim) * chi2_value / eps_ESS ** 2
        marginal_ESS = np.empty((dim, ))
        min_marginal_ESS = np.empty((dim,))
        for j in range(dim):
            marginal_ESS[j] = nsamples * Omega[j,j] / Sigma[j,j]
            min_marginal_ESS[j] = 4 * norm.ppf(alpha_ESS/2)**2 / eps_ESS**2

        print('Univariate Effective Sample Size in each dimension:')
        for j in range(dim):
            print('Dimension {}: ESS = {}, minimum ESS recommended = {}'.
                  format(j+1, marginal_ESS[j], min_marginal_ESS[j]))
        #print('\nMultivariate Effective Sample Size:')
        #print('Multivariate ESS = {}, minimum ESS recommended = {}'.format(joint_ESS, min_joint_ESS))

        # Computation of the autocorrelation time in each dimension
        #def auto_window(taus, c):    # Automated windowing procedure following Sokal (1989)
        #    m = np.arange(len(taus)) < c * taus
        #    if np.any(m):
        #        return np.argmin(m)
        #    return len(taus) - 1
        #autocorrelation_time = []
        #for j in range(samples.shape[1]):
        #    x = samples[:, j] - np.mean(samples[:, j])
        #    f = np.correlate(x, x, mode="full") / np.dot(x, x)
        #    maxlags = len(x) - 1
        #    taus = np.arange(-maxlags, maxlags + 1)
        #    f = f[len(x) - 1 - maxlags:len(x) + maxlags]
        #    window = auto_window(taus, c=5.)
        #    autocorrelation_time.append(taus[window])
        #print('Autocorrelation time in each dimension (for nsamples = ):')
        #for j in range(dim):
        #    print('Dimension {}: autocorrelation time = {}'.format(j+1, autocorrelation_time[j]))

        # Output plots
        if graphics:
            if dim >= 5:
                print('No graphics when dim >= 5')
            else:
                if figsize is None:
                    figsize = (20, 4 * dim)
                fig, ax = plt.subplots(nrows=dim, ncols=3, figsize=figsize)
                for j in range(samples.shape[1]):
                    ax[j, 0].plot(np.arange(nsamples), samples[:, j])
                    ax[j, 0].set_title('chain - parameter # {}'.format(j+1))
                    ax[j, 1].plot(np.arange(nsamples), np.cumsum(samples[:, j])/np.arange(nsamples))
                    ax[j, 1].set_title('parameter convergence')
                    ax[j, 2].acorr(samples[:, j] - np.mean(samples[:, j]), maxlags=40, normed=True)
                    ax[j, 2].set_title('correlation between samples')
                plt.show(fig)

    elif len(samples.chain) == 3:
        print('No diagnostics for various chains of MCMC are currently supported. \n')

    else:
        return ValueError('Wrong dimensions in samples.')



@contextmanager
def suppress_stdout():
    """ A function to suppress output"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_input_dims(x):
    """
    Check that x is a 2D ndarray.

    **Inputs:**

    :param x: Existing samples
    :type x: ndarray (nsamples, dim)

    """
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except:
            raise TypeError('Input should be provided as a nested list of 2d ndarray of shape (nsamples, dimension).')
    if len(x.shape) != 2:
        raise TypeError('Input should be provided as a nested list of 2d ndarray of shape (nsamples, dimension).')
    return x


# Grassmann: svd
def svd(matrix, value):
    """
    Compute the singular value decomposition of a matrix and truncate it.

    Given a matrix compute its singular value decomposition (SVD) and given a desired rank you
    can truncate the matrix containing the eigenvectors.

    **Input:**

    :param matrix: Input matrix.
    :type  matrix: list or numpy array

    :param value: Rank.
    :type  value: int

    **Output/Returns:**

    :param u: left-singular eigenvectors.
    :type  u: numpy array

    :param u: eigenvalues.
    :type  u: numpy array

    :param v: right-singular eigenvectors.
    :type  v: numpy array
    """
    ui, si, vi = np.linalg.svd(matrix, full_matrices=True,hermitian=False)  # Compute the SVD of matrix
    si = np.diag(si)  # Transform the array si into a diagonal matrix containing the singular values
    vi = vi.T  # Transpose of vi

    # Select the size of the matrices u, s, and v
    # either based on the rank of (si) or on a user defined value
    if value == 0:
        rank = np.linalg.matrix_rank(si)  # increase the number of basis up to rank
        u = ui[:, :rank]
        s = si[:rank, :rank]
        v = vi[:, :rank]

    else:
        u = ui[:, :value]
        s = si[:value, :value]
        v = vi[:, :value]

    return u, s, v

def check_arguments(argv, min_num_matrix, ortho):
    
    """
    Check input arguments for consistency.

    Check the input matrices for consistency given the minimum number of matrices (min_num_matrix) 
    and the boolean varible (ortho) to test the orthogonality.

    **Input:**

    :param argv: Matrices to be tested.
    :type  argv: list of arguments

    :param min_num_matrix: Minimum number of matrices.
    :type  min_num_matrix: int
    
    :param ortho: boolean varible to test the orthogonality.
    :type  ortho: bool

    **Output/Returns:**

    :param inputs: Return the input matrices.
    :type  inputs: numpy array

    :param nargs: Number of matrices.
    :type  nargs: numpy array
    """
        
    # Check the minimum number of matrices involved in the operations
    if type(min_num_matrix) != int:
        raise ValueError('The minimum number of matrices MUST be an integer number!')
    elif min_num_matrix < 1:
        raise ValueError('Number of arguments MUST be larger than or equal to one!')

    # Check if the variable controlling the orthogonalization is boolean
    if type(ortho) != bool:
        raise ValueError('The last argument MUST be a boolean!')

    nargv = len(argv)

    # If the number of provided inputs are zero exit the code
    if nargv == 0:
        raise ValueError('Missing input arguments!')

    # Else if the number of arguments is equal to 1 
    elif nargv == 1:

        # Check if the number of expected matrices are higher than or equal to 2
        args = argv[0]
        nargs = len(args)
      
        if np.shape(args)[0] == 1 or len(np.shape(args)) == 2:
            nargs = 1
        # if it is lower than two exit the code, otherwise store them in a list
        if nargs < min_num_matrix:
            raise ValueError('The number of points must be higher than:', min_num_matrix)

        else:
            inputs = []
            if nargs == 1:
                inputs = [args]
            else:

                # Loop over all elements
                for i in range(nargs):                  
                    # Verify the type of the input variables and store in a list
                    inputs.append(test_type(args[i], ortho))

    else:

        nargs = nargv
        # Each argument MUST be a matrix
        inputs = []
        for i in range(nargv):
            # Verify the type of the input variables and store in a list
            inputs.append(test_type(argv[i], ortho))

    return inputs, nargs


def test_type(X, ortho):
    
    """
    Test the datatype of X.

    Check if the datatype of the matrix X is consistent.

    **Input:**

    :param X: Matrices to be tested.
    :type  X: list or numpy array
    
    :param ortho: boolean varible to test the orthogonality.
    :type  ortho: bool

    **Output/Returns:**

    :param Y: Tested and adjusted matrices.
    :type  Y: numpy array
    """
        
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError('Elements of input arguments should be provided either as list or array')
    elif type(X) == list:
        Y = np.array(X)
    else:
        Y = X

    if ortho:
        Ytest = np.dot(Y.T, Y)
        if not np.array_equal(Ytest, np.identity(np.shape(Ytest)[0])):
            Y, unused = np.linalg.qr(Y)

    return Y

def nn_coord(x, k):
    
    """
    Select k elements close to x.

    Select k elements close to x to be used to construct a sparse kernel
    matrix to be used in the diffusion maps.

    **Input:**

    :param x: Matrices to be tested.
    :type  x: list or numpy array
    
    :param k: Number of points close to x.
    :type  k: int

    **Output/Returns:**

    :param idx: Indices of the closer points.
    :type  idx: int
    """
        
    if isinstance(x, list):
        x = np.array(x)
        
    dim = np.shape(x)
    
    if len(dim) is not 1:
        raise ValueError('k MUST be a vector.')
    
    if not isinstance(k, int):
        raise TypeError('k MUST be integer.')

    if k<1:
        raise ValueError('k MUST be larger than or equal to 1.')
    
    #idx = x.argsort()[::-1][:k]
    idx = x.argsort()[:len(x)-k]
    #idx = idx[0:k]
    #idx = idx[k+1:]
    return idx
