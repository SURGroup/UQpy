import itertools
import math

import numpy as np
import scipy.integrate as integrate
from beartype import beartype

from UQpy.distributions import *


class Polynomials:
    """
    Class for polynomials used for the polynomial_chaos method.

    **Inputs:**

    * **dist_object** ('class'):
        Object from a distribution class.

    * **degree** ('int'):
        Maximum degree of the polynomials.

    **Methods:**
    """
    @beartype
    def __init__(self,
                 distributions,
                 degree: int):
        self.distribution = distributions
        self.degree = degree + 1

    @staticmethod
    def standardize_normal(tensor, mean, std):
        """
        Static method: Standardize data based on the standard normal
        distribution N(0,1).

        **Input:**

        * **x** (`ndarray`)
            Input data generated from a normal distribution.

        * **mean** (`list`)
            Mean value of the original normal distribution.

        * **std** (`list`)
            Standard deviation of the original normal distribution.

        **Output/Returns:**

        `ndarray`
            Standardized data.

        """
        return (tensor - mean) / std

    @staticmethod
    def standardize_uniform(tensor, m, scale):
        """
        Static method: Standardize data based on the uniform distribution
        U(-1,1).

        **Input:**

        * **x** (`ndarray`)
            Input data generated from a normal distribution.

        * **m** (`float`)
            Mean value of the original uniform distribution.

        * **b** (`list`)
            Scale of the original uniform distribution.

        **Output/Returns:**

        `ndarray`
            Standardized data.

        """
        return (tensor - m) / (scale / 2)

    @staticmethod
    def normalized(degree, samples, a, b, pdf_st, p):
        """
        Static method: Calculates design matrix and normalized polynomials.

        **Input:**

        * **x** (`ndarray`)
            Input samples.

        * **a** (`float`)
            Left bound of the support the distribution.

        * **b** (`floar`)
            Right bound of the support of the distribution.

        * **pdf_st** (`function`)
            Pdf function generated from UQpy distribution object.

        * **p** (`list`)
            List containing the orthogonal polynomials generated with scipy.

        **Output/Returns:**

        * **a** (`ndarray`)
            Returns the design matrix

        * **pol_normed** (`ndarray`)
            Returns the normalized polynomials.

        """

        pol_normed = []
        m = np.zeros((degree, degree))
        for i in range(degree):
            for j in range(degree):
                int_res = integrate.quad(lambda k: p[i](k) * p[j](k) * pdf_st(k),
                                         a, b, epsabs=1e-15, epsrel=1e-15)
                m[i, j] = int_res[0]
            pol_normed.append(p[i] / np.sqrt(m[i, i]))

        a = np.zeros((samples.shape[0], degree))
        for i in range(samples.shape[0]):
            for j in range(degree):
                a[i, j] = pol_normed[j](samples[i])

        return a, pol_normed

    def get_mean(self):
        """
        Returns a `float` with the mean of the UQpy distribution object.
        """
        m = self.distribution.moments(moments2return='m')
        return m

    def get_std(self):
        """
        Returns a `float` with the variance of the UQpy distribution object.
        """
        s = np.sqrt(self.distribution.moments(moments2return='v'))
        return s

    def location(self):
        """
        Returns a `float` with the location of the UQpy distribution object.
        """
        m = self.distribution.__dict__['parameter_vector']['location']
        return m

    def scale(self):
        """
        Returns a `float` with the scale of the UQpy distribution object.
        """
        s = self.distribution.__dict__['parameter_vector']['scale']
        return s

    def evaluate(self, x):
        """
        Calculates the design matrix. Rows represent the input samples and
        columns the multiplied polynomials whose degree must not exceed the
        maximum degree of polynomials.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the samples.

        **Outputs:**

        * **design** (`ndarray`):
            Returns an array with the design matrix.
        """
        if not type(self.distribution) == JointIndependent:
            if type(self.distribution) == Normal:
                from UQpy.surrogates.polynomial_chaos.polynomials.Hermite import Hermite
                return Hermite(self.degree, self.distribution).get_polys(x)[0]
                # design matrix (second_order_tensor x polynomials)

            if type(self.distribution) == Uniform:
                from UQpy.surrogates.polynomial_chaos.polynomials.Legendre import Legendre
                return Legendre(self.degree, self.distribution).get_polys(x)[0]

            else:
                raise TypeError('Warning: This distribution is not supported.')

        else:

            a = []
            for i in range(len(self.distribution.marginals)):

                if isinstance(self.distribution.marginals[i], Normal):
                    from UQpy.surrogates.polynomial_chaos.polynomials.Hermite import Hermite
                    a.append(Hermite(self.degree,
                                     self.distribution.marginals[i]).get_polys(x[:, i])[0])

                elif isinstance(self.distribution.marginals[i], Uniform):
                    from UQpy.surrogates.polynomial_chaos.polynomials.Legendre import Legendre
                    a.append(Legendre(self.degree,
                                      self.distribution.marginals[i]).get_polys(x[:, i])[0])

                else:
                    raise TypeError('Warning: This distribution is not supported.')

            # Compute all possible valid combinations
            m = len(a)  # number of variables
            p = self.degree  # maximum polynomial order

            p_ = np.arange(0, p, 1).tolist()
            res = list(itertools.product(p_, repeat=m))
            # sum of poly orders
            sum_ = [int(math.fsum(res[i])) for i in range(len(res))]
            indices = sorted(range(len(sum_)), key=lambda k: sum_[k])
            res_new = [res[indices[i]] for i in range(len(res))]
            comb = [(0,) * m]

            for i in range(m):
                t = [0] * m
                t[i] = 1
                comb.append(tuple(t))

            for i in range(len(res_new)):
                if 1 < int(math.fsum(res_new[i])) <= p - 1:
                    rev = res_new[i][::-1]
                    comb.append(rev)

            design = np.ones((x.shape[0], len(comb)))
            for i in range(len(comb)):
                for j in range(m):
                    h = [a[j][k][comb[i][j]] for k in range(x.shape[0])]
                    design[:, i] *= h

            return design
