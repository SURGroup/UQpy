from abc import abstractmethod
from typing import Union, Callable

import numpy as np
import scipy.integrate as integrate
from beartype import beartype
from scipy import stats as stats
from UQpy.distributions.baseclass import Distribution
import warnings

from UQpy.distributions.collection import Uniform, Normal

warnings.filterwarnings('ignore')


class Polynomials:

    @beartype
    def __init__(self, distributions: Union[Distribution, list[Distribution]], degree: int):
        """
        Class for polynomials used for the polynomial_chaos method.

        :param distributions: Object from a distribution class.
        :param degree: Maximum degree of the polynomials.
        """
        self.distributions = distributions
        self.degree = degree + 1

    @staticmethod
    def standardize_sample(x, joint_distribution):
        """
        Static method: Standardize data based on the joint probability distribution.

        :param x: Input data generated from a joint probability distribution.
        :param joint_distribution: joint probability distribution from :py:mod:`UQpy` distribution object
        :return: Standardized data.
        """

        s = np.zeros(x.shape)
        inputs_number = len(x[0, :])
        if inputs_number == 1:
            marginals = [joint_distribution]
        else:
            marginals = joint_distribution.marginals

        for i in range(inputs_number):
            if type(marginals[i]) == Normal:
                s[:, i] = Polynomials.standardize_normal(x[:, i], mean=marginals[i].parameters['loc'],
                                                         std=marginals[i].parameters['scale'])
            elif type(marginals[i]) == Uniform:
                s[:, i] = Polynomials.standardize_uniform(x[:, i], marginals[i])
            else:
                raise TypeError("standarize_sample is defined only for Uniform and Gaussian marginal distributions")
        return s

    @staticmethod
    def standardize_pdf(x, joint_distribution):
        """
        Static method: PDF of standardized distributions associated to Hermite or Legendre polynomials.

        :param x: Input data generated from a joint probability distribution
        :param joint_distribution: joint probability distribution from :py:mod:`UQpy` distribution object
        :return: Value of standardized PDF calculated for x
        """

        inputs_number = len(x[0, :])
        pdf_val = 1
        s = Polynomials.standardize_sample(x, joint_distribution)

        if inputs_number == 1:
            marginals = [joint_distribution]
        else:
            marginals = joint_distribution.marginals

        for i in range(inputs_number):
            if type(marginals[i]) == Normal:
                pdf_val *= (stats.norm.pdf(s[:, i]))
            elif type(marginals[i]) == Uniform:
                pdf_val *= (stats.uniform.pdf(s[:, i], loc=-1, scale=2))
            else:
                raise TypeError("standardize_pdf is defined only for Uniform and Gaussian marginal distributions")
        return pdf_val

    @staticmethod
    def standardize_normal(tensor: np.ndarray, mean: float, std: float):
        """
        Static method: Standardize data based on the standard normal distribution :math:`\mathcal{N}(0,1)`.

        :param tensor: Input data generated from a normal distribution.
        :param mean: Mean value of the original normal distribution.
        :param std: Standard deviation of the original normal distribution.
        :return: Standardized data.
        """
        return (tensor - mean) / std

    @staticmethod
    def standardize_uniform(x, uniform):
        loc = uniform.get_parameters()['loc']  # loc = lower bound of uniform distribution
        scale = uniform.get_parameters()['scale']
        upper = loc + scale  # upper bound = loc + scale
        return (2 * x - loc - upper) / (upper - loc)

    @staticmethod
    def normalized(degree: int, samples: np.ndarray, a: float, b: float, pdf_st: Callable, p: list):
        """
        Calculates design matrix and normalized polynomials.

        :param degree: polynomial degree
        :param samples:  Input samples.
        :param a: Left bound of the support the distribution.
        :param b: Right bound of the support of the distribution.
        :param pdf_st: Pdf function generated from :py:mod:`UQpy` distribution object.
        :param p: List containing the orthogonal polynomials generated with scipy.
        :return: Design matrix,normalized polynomials
        """
        pol_normed = []
        m = np.zeros((degree, degree))
        for i in range(degree):
            for j in range(degree):
                int_res = integrate.quad(
                    lambda k: p[i](k) * p[j](k) * pdf_st(k),
                    a,
                    b,
                    epsabs=1e-15,
                    epsrel=1e-15,
                )
                m[i, j] = int_res[0]
            pol_normed.append(p[i] / np.sqrt(m[i, i]))

        a = np.zeros((samples.shape[0], degree))
        for i in range(samples.shape[0]):
            for j in range(degree):
                a[i, j] = pol_normed[j](samples[i])

        return a, pol_normed

    def get_mean(self):
        """
        Returns a :any:`float` with the mean of the :py:mod:`UQpy` distribution object.
        """
        m = self.distributions.moments(moments2return="m")
        return m

    def get_std(self):
        """
        Returns a :any:`float` with the variance of the :py:mod:`UQpy` distribution object.
        """
        s = np.sqrt(self.distributions.moments(moments2return="v"))
        return s

    def location(self):
        """
        Returns a :any:`float` with the location of the :py:mod:`UQpy` distribution object.
        """
        m = self.distributions.__dict__["parameters"]["location"]
        return m

    def scale(self):
        """
        Returns a :any:`float` with the scale of the :py:mod:`UQpy` distribution object.
        """
        s = self.distributions.__dict__["parameters"]["scale"]
        return s

    @abstractmethod
    def evaluate(self, x: np.ndarray):
        pass

    distribution_to_polynomial = {}
