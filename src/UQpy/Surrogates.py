import chaospy as cp
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from UQpy.Distributions import *


########################################################################################################################
########################################################################################################################
#                                         Stochastic Reduced Order Model  (SROM)                                       #
########################################################################################################################
########################################################################################################################

class SROM:

    def __init__(self, samples=None, pdf_type=None, moments=None, weights_errors=None,
                 weights_distribution=None, weights_moments=None, weights_correlation=None,
                 properties=None, pdf_params=None, correlation=None):
        """
        Stochastic Reduced Order Model(SROM) provide a low-dimensional, discrete approximation of a given random
        quantity.

        SROM generates a discrete approximation of continuous random variables. The probabilities/weights are
        considered to be the parameters for the SROM and they can be obtained by minimizing the error between the
        marginal distributions, first and second order moments about origin and correlation between random variables.

        References:
        M. Grigoriu, "Reduced order models for random functions. Application to stochastic problems",
            Applied Mathematical Modelling, Volume 33, Issue 1, Pages 161-175, 2009.

        Input:
        :param samples: A list of samples corresponding to each random variables
        :type samples: list

        :param pdf_type: A list of Cumulative distribution functions of random variables
        :type pdf_type: list str or list function

        :param pdf_params: Parameters of distribution
        :type pdf_params: list

        :param moments: A list containing first and second order moment about origin of all random variables
        :type moments: list

        :param weights_errors: Weights associated with error in distribution, moments and correlation.
                               Default: weights_errors = [1, 0.2, 0]
        :type weights_errors: list or array

        :param properties: A list of booleans representing properties, which are required to match in reduce
                           order model. This class focus on reducing errors in distribution, first order moment
                           about origin, second order moment about origin and correlation of samples.
                           Default: properties = [True, True, True, False]
                           Example: properties = [True, True, False, False] will minimize errors in distribution and
                           errors in first order moment about origin in reduce order model.
        :type properties: list

        :param weights_distribution: An list or array containing weights associated with different samples.
                                     Options:
                                        If weights_distribution is None, then default value is assigned.
                                        If size of weights_distribution is 1xd, then it is assigned as dot product
                                            of weights_distribution and default value.
                                        Otherwise size of weights_distribution should be equal to Nxd.
                                     Default: weights_distribution = Nxd dimensional array with all elements equal
                                     to 1.
        :type weights_distribution: ndarray or list

        :param weights_moments: An array of dimension 2xd, where 'd' is number of random variables. It contain
                                weights associated with moments.
                                Options:
                                    If weights_moments is None, then default value is assigned.
                                    If size of weights_moments is 1xd, then it is assigned as dot product
                                        of weights_moments and default value.
                                    Otherwise size of weights_distribution should be equal to 2xd.
                                Default: weights_moments = Square of reciprocal of elements of moments.
        :type weights_moments: ndarray or list (float)

        :param weights_correlation: An array of dimension dxd, where 'd' is number of random variables. It contain
                                    weights associated with correlation of random variables.
                                    Default: weights_correlation = dxd dimensional array with all elements equal to
                                    1.
        :type weights_correlation: ndarray or list

        :param correlation: Correlation matrix between random variables.
        :type correlation: list


        Output:
        :return: SROM.samples: Last column contains the probabilities/weights defining discrete approximation of
                               continuous random variables.
        :rtype: SROM.samples: ndarray
        """
        # Authors: Mohit Chauhan
        # Updated: 5/12/18 by Mohit Chauhan

        self.samples = np.array(samples)
        self.correlation = np.array(correlation)
        self.pdf_type = pdf_type
        self.moments = np.array(moments)
        self.weights_errors = weights_errors
        self.weights_distribution = weights_distribution
        self.weights_moments = weights_moments
        self.weights_correlation = weights_correlation
        self.properties = properties
        self.pdf_params = pdf_params
        self.dimension = samples.shape[1]
        self.nsamples = samples.shape[0]
        self.init_srom()
        self.sample_weights = self.run_srom()

    def run_srom(self):
        from scipy import optimize

        def f(p_, samples, wd, wm, wc, mar, n, d, m, alpha, para, prop, correlation):
            e1 = 0.
            e2 = 0.
            e22 = 0.
            e3 = 0.
            com = np.append(samples, np.transpose(np.matrix(p_)), 1)
            for j in range(d):
                srt = com[np.argsort(com[:, j].flatten())]
                s = srt[0, :, j]
                a = srt[0, :, d]
                A = np.cumsum(a)
                marginal = mar[j]

                if prop[0] is True:
                    for i in range(n):
                        e1 += wd[i, j] * (A[0, i] - marginal(s[0, i], para[j])) ** 2

                if prop[1] is True:
                    e2 += wm[0, j] * (np.sum(np.array(p_) * samples[:, j]) - m[0, j]) ** 2

                if prop[2] is True:
                    e22 += wm[1, j] * (
                            np.sum(np.array(p_) * (samples[:, j] * samples[:, j])) - m[1, j]) ** 2

                if prop[3] is True:
                    for k in range(d):
                        if k > j:
                            r = correlation[j, k] * np.sqrt((m[1, j] - m[0, j] ** 2) * (m[1, k] - m[0, k] ** 2)) + \
                                m[0, j] * m[0, k]
                            e3 += wc[k, j] * (
                                    np.sum(np.array(p_) * (
                                                np.array(samples[:, j]) * np.array(samples[:, k]))) - r) ** 2

            return alpha[0] * e1 + alpha[1] * (e2 + e22) + alpha[2] * e3

        def constraint(x):
            return np.sum(x) - 1

        def constraint2(y):
            n = np.size(y)
            return np.ones(n) - y

        def constraint3(z):
            n = np.size(z)
            return z - np.zeros(n)

        cons = ({'type': 'eq', 'fun': constraint}, {'type': 'ineq', 'fun': constraint2},
                {'type': 'ineq', 'fun': constraint3})

        p_ = optimize.minimize(f, np.zeros(self.nsamples),
                               args=(self.samples, self.weights_distribution, self.weights_moments,
                                     self.weights_correlation, self.pdf_type, self.nsamples, self.dimension,
                                     self.moments, self.weights_errors, self.pdf_params, self.properties,
                                     self.correlation),
                               constraints=cons, method='SLSQP')

        return p_.x

    def init_srom(self):

        if self.pdf_type is None:
            raise NotImplementedError("Exit code: Distribution not defined.")

        self.pdf_type = cdf(self.pdf_type)

        # Check samples
        if self.samples is None:
            raise NotImplementedError('Samples not provided for SROM')

        # Check properties to match
        if self.properties is None:
            self.properties = [True, True, True, False]

        # Check moments and correlation
        if self.properties[1] is True or self.properties[2] is True or self.properties[3] is True:
            if self.moments is None:
                raise NotImplementedError("'moments' are required")
        # Both moments are required, if correlation property is required to be match
        if self.properties[3] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("1. Size of 'moments' is not correct")
            if self.correlation is None:
                self.correlation = np.identity(self.dimension)
        # moments.shape[0] should be 1 or 2
        if self.moments.shape != (1, self.dimension) and self.moments.shape != (2, self.dimension):
            raise NotImplementedError("2. Size of 'moments' is not correct")
        # If both the moments are to be included in objective function, then moments.shape[0] should be 2
        if self.properties[1] is True and self.properties[2] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("3. Size of 'moments' is not correct")
        # If only second order moment is to be included in objective function and moments.shape[0] is 1. Then
        # self.moments is converted shape = (2, self.dimension) where is second row contain second order moments.
        if self.properties[1] is False and self.properties[2] is True:
            if self.moments.shape == (1, self.dimension):
                temp = np.ones(shape=(1, self.dimension))
                self.moments = np.concatenate((temp, self.moments))

        # Check weights corresponding to errors
        if self.weights_errors is None:
            self.weights_errors = [1, 0.2, 0]
        self.weights_errors = np.array(self.weights_errors).astype(np.float64)

        # Check weights corresponding to distribution
        if self.weights_distribution is None or not self.weights_distribution:
            self.weights_distribution = np.ones(shape=(self.samples.shape[0], self.dimension))

        self.weights_distribution = np.array(self.weights_distribution)
        if self.weights_distribution.shape == (1, self.dimension):
            self.weights_distribution = self.weights_distribution * np.ones(shape=(self.samples.shape[0],
                                                                                   self.dimension))
        elif self.weights_distribution.shape != (self.samples.shape[0], self.dimension):
            raise NotImplementedError("Size of 'weights for distribution' is not correct")

        # Check weights corresponding to moments and it's default list
        if self.weights_moments is None or not self.weights_moments:
            self.weights_moments = np.reciprocal(np.square(self.moments))

        self.weights_moments = np.array(self.weights_moments)
        if self.weights_moments.shape == (1, self.dimension):
            self.weights_moments = self.weights_moments * np.ones(shape=(2, self.dimension))
        elif self.weights_moments.shape != (2, self.dimension):
            raise NotImplementedError("Size of 'weights for moments' is not correct")

        # Check weights corresponding to correlation and it's default list
        if self.weights_correlation is None or not self.weights_correlation:
            self.weights_correlation = np.ones(shape=(self.dimension, self.dimension))

        self.weights_correlation = np.array(self.weights_correlation)
        if self.weights_correlation.shape != (self.dimension, self.dimension):
            raise NotImplementedError("Size of 'weights for correlation' is not correct")

        # Check cdf_type
        if len(self.pdf_type) == 1:
            self.pdf_type = self.pdf_type * self.dimension
            self.pdf_params = [self.pdf_params] * self.dimension
        elif len(self.pdf_type) != self.dimension:
            raise NotImplementedError("Size of cdf_type should be 1 or equal to dimension")


class SurrogateModels:
    """
    A class containing various surrogate models

    """

########################################################################################################################
########################################################################################################################
#                                        Polynomial Chaos
########################################################################################################################
    class PolynomialChaos:

        """
        A class used to generate a Polynomial Chaos surrogate.

        :param dimension: Dimension of the input space
        :param input:     Input data of shape:  (N x Dimension)
        :param output:    Output data of shape:  (N,)
        :param order:     Order of the polynomial chaos model


        Created by: Dimitris G. Giovanis
        Last modified: 12/10/2017
        Last modified by: Dimitris G. Giovanis

        """

        def __init__(self, dimension=None, input=None, output=None, order=None):

            self.dimension = dimension
            self.input = np.transpose(input)
            self.output = output
            self.order = order

            self.distribution = cp.Iid(cp.Uniform(0, 1), self.dimension)
            orthogonal_expansion = cp.orth_ttr(self.order, self.distribution)
            self.poly = cp.fit_regression(orthogonal_expansion, self.input, self.output)

        def PCpredictor(self, sample):

            if len(sample.shape) == 1:
                g_tilde = 0.0
                if self.dimension == 1:
                    g_tilde = self.poly(sample[0])

                elif self.dimension == 2:
                    g_tilde = self.poly(sample[0], sample[1])

            else:
                g_tilde = np.zeros(sample.shape[0])
                for i in range(sample.shape[0]):
                    if self.dimension == 1:
                        g_tilde[i] = self.poly(sample[i])

                    elif self.dimension == 2:
                        g_tilde[i] = self.poly(sample[i][0], sample[i][1])
                        print()

            return g_tilde


########################################################################################################################
########################################################################################################################
#                                        Gaussian Process
########################################################################################################################
    class GaussianProcess:

        """
        A class used to generate a Gaussian process surrogate.

        :param input:  Input data of shape:  (N x Dimension)
        :param output: Output data of shape:  (N,)


        Created by: Dimitris G. Giovanis
        Last modified: 12/10/2017
        Last modified by: Dimitris G. Giovanis

        """

        def __init__(self, input=None, output=None):

            self.input = input
            self.output = output

            kernel = C(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-3, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
            gp.fit(self.input, self.output)
            self.gp = gp

        def GPredictor(self, sample):

            if len(sample.shape) == 1:
                sample = sample.reshape(-1, 1)
                g_tilde, g_std = self.gp.predict(sample.T, return_std=True)
                return g_tilde[0], g_std[0]
            else:
                g_tilde, g_std = self.gp.predict(sample, return_std=True)
                return g_tilde, g_std
