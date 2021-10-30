import scipy.special as special
from beartype import beartype

from UQpy.distributions import Uniform
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import (
    Polynomials,
)


class Legendre(Polynomials):

    @beartype
    def __init__(self, degree: int, distributions):
        """
        Class of univariate polynomials appropriate for data generated from a uniform distribution.

        :param degree: Maximum degree of the polynomials.
        :param distributions: Distribution object of the generated samples.
        """
        super().__init__(distributions, degree)
        self.degree = degree
        self.pdf = self.distribution.pdf

    def get_polys(self, x):
        """
        Calculates the normalized Legendre polynomials evaluated at sample points.

        :param x: `ndarray` containing the samples.
        :return: Α list of 'ndarrays' with the design matrix and the
                    normalized polynomials.
        """
        a, b = -1, 1
        m, scale = Polynomials.get_mean(self), Polynomials.scale(self)
        x_ = Polynomials.standardize_uniform(x, m, scale)

        uni = Uniform(a, b - a)
        pdf_st = uni.pdf

        p = []
        for i in range(self.degree):
            p.append(special.legendre(i, monic=False))

        return Polynomials.normalized(self.degree, x_, a, b, pdf_st, p)
