from typing import Union

import numpy as np
import scipy.special as special
from beartype import beartype

from UQpy.distributions import Uniform
from UQpy.distributions.baseclass import Distribution
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials
from scipy.special import eval_legendre
import math


class Legendre(Polynomials):

    @beartype
    def __init__(self, degree: int, distributions: Union[Distribution, list[Distribution]]):
        """
        Class of univariate polynomials appropriate for data generated from a uniform distribution.

        :param degree: Maximum degree of the polynomials.
        :param distributions: Distribution object of the generated samples.
        """
        super().__init__(distributions, degree)
        self.degree = degree
        self.pdf = self.distributions

    def evaluate(self, x: np.ndarray):
        """
        Calculates the normalized Legendre polynomials evaluated at sample points.

        :param x: :class:`numpy.ndarray` containing the samples.
        :return: Α list of :class:`numpy.ndarray` with the design matrix and the
                    normalized polynomials.
        """
        x = np.array(x).flatten()

        # normalize data
        x_normed = Polynomials.standardize_uniform(x, self.distributions)

        # evaluate standard Legendre polynomial, i.e. orthogonal in [-1,1] with
        # PDF = 1 (NOT 1/2!!!)
        l = eval_legendre(self.degree, x_normed)

        # normalization constant
        st_lege_norm = np.sqrt(2 / (2 * self.degree + 1))

        # multiply by sqrt(2) to take into account the pdf 1/2
        l = np.sqrt(2) * l / st_lege_norm

        return l

    @staticmethod
    def legendre_triple_product(k, l, m):

        normk = 1 / ((2 * k) + 1)
        norml = 1 / ((2 * l) + 1)
        normm = 1 / ((2 * m) + 1)
        norm = np.sqrt(normm / (normk * norml))

        return norm * (2 * m + 1) * Legendre.wigner_3j_PCE(k, l, m) ** 2

    @staticmethod
    def wigner_3j_PCE(j_1, j_2, j_3):

        cond1 = j_1 + j_2 - j_3
        cond2 = j_1 - j_2 + j_3
        cond3 = -j_1 + j_2 + j_3
        if cond1 < 0 or cond2 < 0 or cond3 < 0:
            return 0
        else:

            factarg = (math.factorial(j_1 + j_2 - j_3) * math.factorial(j_1 - j_2 + j_3) *
                       math.factorial(-j_1 + j_2 + j_3) * math.factorial(j_1) ** 2 * math.factorial(j_2) ** 2 *
                       math.factorial(j_3) ** 2) / math.factorial(j_1 + j_2 + j_3 + 1)

            factfinal = np.sqrt(factarg)

            imin = max(-j_3 + j_1, -j_3 + j_2, 0)
            imax = min(j_2, j_1, j_1 + j_2 - j_3)
            summfinal = 0

            for i in range(imin, imax + 1):
                sumfact = math.factorial(i) * \
                          math.factorial(i + j_3 - j_1) * \
                          math.factorial(j_2 - i) * \
                          math.factorial(j_1 - i) * \
                          math.factorial(i + j_3 - j_2) * \
                          math.factorial(j_1 + j_2 - j_3 - i)
                summfinal = summfinal + int((-1) ** i) / sumfact

            const1 = int((-1) ** int(j_1 - j_2))

            return factfinal * summfinal * const1

Polynomials.distribution_to_polynomial[Uniform] = Legendre
