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


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from contextlib import contextmanager
import sys
import os
from scipy.special import gamma
from scipy.stats import chi2, norm


def transform_ng_to_g(corr_norm, dist, dist_params, samples_ng, jacobian=True):

    """
        Description:

            A function that performs transformation of a non-Gaussian random variable to a Gaussian one.

        Input:
            :param corr_norm: Correlation matrix in the standard normal space
            :type corr_norm: ndarray

            :param dist: marginal distributions
            :type dist: list

            :param dist_params: marginal distribution parameters
            :type dist_params: list

            :param samples_ng: non-Gaussian samples
            :type samples_ng: ndarray

            :param jacobian: The Jacobian of the transformation
            :type jacobian: ndarray

        Output:
            :return: samples_g: Gaussian samples
            :rtype: samples_g: ndarray

            :return: jacobian: The jacobian
            :rtype: jacobian: ndarray

    """

    from scipy.linalg import cholesky

    a_ = cholesky(corr_norm, lower=True)
    samples_g = np.zeros_like(samples_ng)
    m, n = np.shape(samples_ng)
    for j in range(n):
        cdf = dist[j].cdf
        samples_g[:, j] = stats.norm.ppf(cdf(samples_ng[:, j], dist_params[j]))

    if not jacobian:
        print("UQpy: Done.")
        return samples_g, None
    else:
        temp_ = np.zeros([n, n])
        jacobian = [None] * m
        for i in range(m):
            for j in range(n):
                pdf = dist[j].pdf
                temp_[j, j] = stats.norm.pdf(samples_g[i, j]) / pdf(samples_ng[i, j], dist_params[j])
            jacobian[i] = np.linalg.solve(temp_, a_)

        return samples_g, jacobian


def transform_g_to_ng(corr_norm, dist, dist_params, samples_g, jacobian=True):

    """
        Description:

            A function that performs transformation of a Gaussian random variable to a non-Gaussian one.

        Input:
            :param corr_norm: Correlation matrix in the standard normal space
            :type corr_norm: ndarray

            :param dist: marginal distributions
            :type dist: list

            :param dist_params: marginal distribution parameters
            :type dist_params: list

            :param samples_g: Gaussian samples
            :type samples_g: ndarray

            :param jacobian: The Jacobian of the transformation
            :type jacobian: ndarray

        Output:
            :return: samples_ng: Gaussian samples
            :rtype: samples_ng: ndarray

            :return: jacobian: The jacobian
            :rtype: jacobian: ndarray

    """

    from scipy.linalg import cholesky

    samples_ng = np.zeros_like(samples_g)
    m, n = np.shape(samples_g)
    for j in range(n):
        i_cdf = dist[j].icdf
        samples_ng[:, j] = i_cdf(stats.norm.cdf(samples_g[:, j]), dist_params[j])

    if not jacobian:
        print("UQpy: Done.")
        return samples_ng, None
    else:
        a_ = cholesky(corr_norm, lower=True)
        temp_ = np.zeros([n, n])
        jacobian = [None] * m
        for i in range(m):
            for j in range(n):
                pdf = dist[j].pdf
                temp_[j, j] = pdf(samples_ng[i, j], dist_params[j]) / stats.norm.pdf(samples_g[i, j])
            jacobian[i] = np.linalg.solve(a_, temp_)

        return samples_ng, jacobian


def run_corr(samples, corr):

    """
        Description:

            A function which performs the Cholesky decomposition of the correlation matrix and correlates standard
            normal samples.

        Input:
            :param corr: Correlation matrix
            :type corr: ndarray

            :param samples: Standard normal samples.
            :type samples: ndarray


        Output:
            :return: samples_corr: Correlated standard normal samples
            :rtype: samples_corr: ndarray

    """

    from scipy.linalg import cholesky
    c = cholesky(corr, lower=True)
    samples_corr = np.dot(c, samples.T)

    return samples_corr.T


def run_decorr(samples, corr):

    """
        Description:

            A function which performs the Cholesky decomposition of the correlation matrix and de-correlates standard
            normal samples.

        Input:
            :param corr: Correlation matrix
            :type corr: ndarray

            :param samples: standard normal samples.
            :type samples: ndarray


        Output:
            :return: samples_uncorr: Uncorrelated standard normal samples
            :rtype: samples_uncorr: ndarray

    """

    from scipy.linalg import cholesky

    c = cholesky(corr, lower=True)
    inv_corr = np.linalg.inv(c)
    samples_uncorr = np.dot(inv_corr, samples.T)

    return samples_uncorr.T


def correlation_distortion(marginal, params, rho_norm):

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
    for i in range(len(marginal)):
        i_cdf_i = marginal[i].icdf
        moments_i = marginal[i].moments
        mi = moments_i(params[i])
        if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
            raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

        for j in range(i + 1, len(marginal)):
            i_cdf_j = marginal[j].icdf
            moments_j = marginal[j].moments
            mj = moments_j(params[j])
            if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

            tmp_f_xi = ((i_cdf_j(stats.norm.cdf(xi), params[j]) - mj[0]) / np.sqrt(mj[1]))
            tmp_f_eta = ((i_cdf_i(stats.norm.cdf(eta), params[i]) - mi[0]) / np.sqrt(mi[1]))
            coef = tmp_f_xi * tmp_f_eta * w2d

            rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
            rho[j, i] = rho[i, j]

    print('UQpy: Done.')
    return rho


def itam(marginal, params, corr, beta, thresh1, thresh2):

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
            :type beta: int

            :param thresh1: Threshold
            :type thresh1: float

            :param thresh2: Threshold
            :type thresh2: float

        Output:
            :return corr_norm: Gaussian correlation matrix
            :rtype corr_norm: ndarray

    """

    if beta is None:
        beta = 1
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
        corr0 = correlation_distortion(marginal, params, corr_norm0)
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


def s_to_r(s, w, t):

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
    r = np.zeros([s.shape[0], len(t)])
    for i in range(s.shape[0]):
        for j in range(len(t)):
            if s.shape[0] == 1:
                r[i, j] = 2 * np.dot(fac, s[i, :] * np.cos(w * t[j]))
            else:
                r[i, j] = 2 * np.dot(fac, np.sqrt((s[i, :] * s[j, :])) * np.cos(w * (t[i] - t[j])))
    return r


def r_to_s(r, w, t):

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

    s = np.zeros([r.shape[0], len(w)])
    for i in range(r.shape[0]):
        for j in range(len(w)):
            r[i, j] = 2 / (2 * np.pi) * np.dot(fac, (r[i, :] * np.cos(t * w[j])))
    s[s < 0] = 0
    return s


def gradient(sample=None, dimension=None, eps=None,  model_script=None, model_object_name=None, input_template=None,
             var_names=None,
             output_script=None, output_object_name=None, ntasks=None, cores_per_task=None, nodes=None, resume=None,
             verbose=None, model_dir=None, cluster=None, order=None):
    """
         Description: A function to estimate the gradients (1st, 2nd, mixed) of a function using finite differences


         Input:
             :param sample: The sample values at which the gradient of the model will be evaluated. Samples can be
             passed directly as  an array or can be passed through the text file 'UQpy_Samples.txt'.
             If passing samples via text file, set samples = None or do not set the samples input.
             :type sample: ndarray

             :param order: The type of derivatives to calculate (1st order, second order, mixed).
             :type order: str

             :param dimension: Number of random variables.
             :type dimension: int

             :param eps: step for the finite difference.
             :type eps: float

             :param model_script: The filename of the Python script which contains commands to execute the model

             :param model_object_name: The name of the function or class which executes the model

             :param input_template: The name of the template input file which will be used to generate input files for
              each run of the model. Refer documentation for more details.

             :param var_names: A list containing the names of the variables which are present in the template input
              files

             :param output_script: The filename of the Python script which contains the commands to process the output

             :param output_object_name: The name of the function or class which has the output values. If the object
              is a class named cls, the output must be saved as cls.qoi. If it a function, it should return the output
              quantity of interest

             :param ntasks: Number of tasks to be run in parallel. RunModel uses GNU parallel to execute models which
              require an input template

             :param cores_per_task: Number of cores to be used by each task

             :param nodes: On MARCC, each node has 24 cores_per_task. Specify the number of nodes if more than one
              node is required.

             :param resume: This option can be set to True if a parallel execution of a model with input template
              failed to finish running all jobs. GNU parallel will then run only the jobs which failed to execute.

             :param verbose: This option can be set to False if you do not want RunModel to print status messages to
              the screen during execution. It is True by default.

             :param model_dir: The directory  that contains the Python script which contains commands to execute the
             model

             :param cluster: This option defines if we run the code into a cluster

         Output:
             :return du_dj: vector of first-order gradients
             :rtype: ndarray
             :return d2u_dj: vector of second-order gradients
             :rtype: ndarray
             :return d2u_dij: vector of mixed gradients
             :rtype: ndarray
     """

    from UQpy.RunModel import RunModel

    if order is None:
        raise ValueError('Exit code: Provide type of derivatives: first, second or mixed.')

    if dimension is None:
     raise ValueError('Error: Dimension must be defined')

    if eps is None:
        eps = [0.1]*dimension
    elif isinstance(eps, float):
        eps = [eps] * dimension
    elif isinstance(eps, list):
        if len(eps) != 1 and len(eps) != dimension:
            raise ValueError('Exit code: Inconsistent dimensions.')
        if len(eps) == 1:
            eps = [eps[0]] * dimension

    if order == 'first' or order == 'second':
        du_dj = np.zeros(dimension)
        d2u_dj = np.zeros(dimension)
        for i in range(dimension):
            x_i1_j = np.array(sample)
            x_i1_j[0, i] += eps[i]
            x_1i_j = np.array(sample)
            x_1i_j[0, i] -= eps[i]

            g0 = RunModel(samples=x_i1_j,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            g1 = RunModel(samples=x_1i_j,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            du_dj[i] = (g0.qoi_list[0] - g1.qoi_list[0])/(2*eps[i])

            if order == 'second':
                g = RunModel(samples=sample, model_script=model_script,
                             model_object_name=model_object_name,
                             input_template=input_template, var_names=var_names, output_script=output_script,
                             output_object_name=output_object_name,
                             ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                             verbose=verbose, model_dir=model_dir, cluster=cluster)

                d2u_dj[i] = (g0.qoi_list[0] - 2 * g.qoi_list[0] + g1.qoi_list[0]) / (eps[i]**2)

        return np.vstack([du_dj, d2u_dj])

    elif order == 'mixed':
        import itertools
        range_ = list(range(dimension))
        d2u_dij = list()
        for i in itertools.combinations(range_, 2):
            x_i1_j1 = np.array(sample)
            x_i1_1j = np.array(sample)
            x_1i_j1 = np.array(sample)
            x_1i_1j = np.array(sample)

            x_i1_j1[0, i[0]] += eps[i[0]]
            x_i1_j1[0, i[1]] += eps[i[1]]

            x_i1_1j[0, i[0]] += eps[i[0]]
            x_i1_1j[0, i[1]] -= eps[i[1]]

            x_1i_j1[0, i[0]] -= eps[i[0]]
            x_1i_j1[0, i[1]] += eps[i[1]]

            x_1i_1j[0, i[0]] -= eps[i[0]]
            x_1i_1j[0, i[1]] -= eps[i[1]]

            g0 = RunModel(samples=x_i1_j1,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            g1 = RunModel(samples=x_i1_1j,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            g2 = RunModel(samples=x_1i_j1,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            g3 = RunModel(samples=x_1i_1j,  model_script=model_script,
                          model_object_name=model_object_name,
                          input_template=input_template, var_names=var_names, output_script=output_script,
                          output_object_name=output_object_name,
                          ntasks=ntasks, cores_per_task=cores_per_task, nodes=nodes, resume=resume,
                          verbose=verbose, model_dir=model_dir, cluster=cluster)

            d2u_dij.append((g0.qoi_list[0] - g1.qoi_list[0] - g2.qoi_list[0] + g3.qoi_list[0])
                           / (4 * eps[i[0]]*eps[i[1]]))

        return np.array(d2u_dij)


def eval_hessian(dimension, mixed_der, der):

    """
    Calculate the hessian matrix with finite differences
    Parameters:

    """
    hessian = np.diag(der)
    import itertools
    range_ = list(range(dimension))
    add_ = 0
    for i in itertools.combinations(range_, 2):
        hessian[i[0], i[1]] = mixed_der[add_]
        hessian[i[1], i[0]] = hessian[i[0], i[1]]
        add_ += 1
    return hessian

def diagnostics(sampling_method, sampling_outputs=None, samples=None, weights=None,
                figsize=None, eps_ESS=0.05, alpha_ESS=0.05):

    """
         Description: A function to estimate the gradients (1st, 2nd, mixed) of a function using finite differences


         Input:
             :param sampling_method: sampling method used to generate samples
             :type sampling_method: str, 'MCMC' or 'IS'

             :param sampling_outputs: output object of a sampling method
             :type sampling_outputs: object of class MCMC or IS

             :param samples: output samples of a sampling method (alternative to giving sampling_outputs for MCMC)
             :type samples: ndarray

             :param weights: output weights of IS (alternative to giving sampling_outputs for IS)
             :type weights: ndarray

             :param figsize: size of the figure for output plots
             :type figsize: tuple (width, height)

             :param eps_ESS: small number required to compute ESS when sampling_method='MCMC', see documentation
             :type eps_ESS: float in [0,1]

             :param alpha_ESS: small number required to compute ESS when sampling_method='MCMC', see documentation
             :type alpha_ESS: float in [0,1]

         Output:
             returns various diagnostics values/plots to evaluate importance sampling and MCMC sampling outputs
     """

    if (eps_ESS < 0) or (eps_ESS > 1):
        raise ValueError('UQpy error: eps_ESS should be a float between 0 and 1.')
    if (alpha_ESS < 0) or (alpha_ESS > 1):
        raise ValueError('UQpy error: alpha_ESS should be a float between 0 and 1.')

    if sampling_method == 'IS':
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
        if figsize is None:
            figsize = (8, 3)
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(weights, np.zeros((np.size(weights), )), s=weights*300, marker='o')
        ax.set_xlabel('weights')
        ax.set_title('Normalized weights out of importance sampling')
        plt.show(fig)

    elif sampling_method == 'MCMC':
        if (sampling_outputs is None) and (samples is None):
            raise ValueError('UQpy error: sampling_outputs or samples should be provided')
        if sampling_outputs is not None:
            samples = sampling_outputs.samples
        print('Diagnostics for MCMC \n')
        nsamples, nparams = samples.shape

        # Acceptance ratio
        if sampling_outputs is not None:
            print('Acceptance ratio of the chain = {}. \n'.format(sampling_outputs.accept_ratio))

        # Computation of ESS and min ESS
        eps = eps_ESS
        alpha = alpha_ESS

        bn = np.ceil(nsamples**(1/2)) # nb of samples per bin
        an = int(np.ceil(nsamples/bn)) # nb of bins, for computation of
        idx = np.array_split(np.arange(nsamples), an)

        means_subdivisions = np.empty((an, samples.shape[1]))
        for i, idx_i in enumerate(idx):
            x_sub = samples[idx_i, :]
            means_subdivisions[i,:] = np.mean(x_sub, axis=0)
        Omega = np.cov(samples.T)
        Sigma = np.cov(means_subdivisions.T)
        joint_ESS = nsamples*np.linalg.det(Omega)**(1/nparams)/np.linalg.det(Sigma)**(1/nparams)
        chi2_value = chi2.ppf(1 - alpha, df=nparams)
        min_joint_ESS = 2 ** (2 / nparams) * np.pi / (nparams * gamma(nparams / 2)) ** (
                    2 / nparams) * chi2_value / eps ** 2
        marginal_ESS = np.empty((nparams, ))
        min_marginal_ESS = np.empty((nparams,))
        for j in range(nparams):
            marginal_ESS[j] = nsamples * Omega[j,j]/Sigma[j,j]
            min_marginal_ESS[j] = 4 * norm.ppf(alpha/2)**2 / eps**2

        print('Univariate Effective Sample Size in each dimension:')
        for j in range(nparams):
            print('Parameter # {}: ESS = {}, minimum ESS recommended = {}'.
                  format(j+1, marginal_ESS[j], min_marginal_ESS[j]))
        print('\nMultivariate Effective Sample Size:')
        print('Multivariate ESS = {}, minimum ESS recommended = {}'.format(joint_ESS, min_joint_ESS))

        # Output plots
        if figsize is None:
            figsize = (20,4*nparams)
        fig, ax = plt.subplots(nrows=nparams, ncols=3, figsize=figsize)
        for j in range(samples.shape[1]):
            ax[j, 0].plot(np.arange(nsamples), samples[:,j])
            ax[j, 0].set_title('chain - parameter # {}'.format(j+1))
            ax[j, 1].plot(np.arange(nsamples), np.cumsum(samples[:,j])/np.arange(nsamples))
            ax[j, 1].set_title('parameter convergence')
            ax[j, 2].acorr(samples[:,j]-np.mean(samples[:,j]), maxlags = 50, normed=True)
            ax[j, 2].set_title('correlation between samples')
        plt.show(fig)

    else:
        raise ValueError('Supported sampling methods for diagnostics are "MCMC", "IS".')
    return fig, ax


def resample(samples, weights, method='multinomial', size=None):
    nsamples = samples.shape[0]
    if size is None:
        size = nsamples
    if method == 'multinomial':
        multinomial_run = np.random.multinomial(size, weights, size=1)[0]
        idx = list()
        for j in range(nsamples):
            if multinomial_run[j] > 0:
                idx.extend([j for _ in range(multinomial_run[j])])
        output = samples[idx, :]
        return output
    else:
        raise ValueError('Exit code: Current available method: multinomial')


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
