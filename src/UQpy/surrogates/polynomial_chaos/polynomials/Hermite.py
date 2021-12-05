from beartype import beartype

from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import (
    Polynomials,
)
import numpy as np
from UQpy.distributions import Normal
import scipy.special as special
from scipy.special import eval_hermitenorm
import math


class Hermite(Polynomials):

    @beartype
    def __init__(self, degree: int, distributions):
        """
        Class of univariate polynomials appropriate for data generated from a normal distribution.

        :param degree: Maximum degree of the polynomials.
        :param distribution: Distribution object of the generated samples.
        """
        super().__init__(distributions, degree)
        self.degree = degree
        self.pdf = self.distributions.pdf

    def get_polys(self, x):
        """
        Calculates the normalized Hermite polynomials evaluated at sample points.

        :param x: `ndarray` containing the samples.
        :return: Α list of 'ndarrays' with the design matrix and the
                    normalized polynomials.
        """
        a, b = -np.inf, np.inf
        mean_ = Polynomials.get_mean(self)
        std_ = Polynomials.get_std(self)
        x_ = Polynomials.standardize_normal(x, mean_, std_)

        norm = Normal(0, 1)
        pdf_st = norm.pdf

        p = []
        for i in range(self.degree):
            p.append(special.hermitenorm(i, monic=False))

        return Polynomials.normalized(self.degree, x_, a, b, pdf_st, p)

    def evaluate(self, x):
        x = np.array(x).flatten()

        # normalize data
        x_normed = Polynomials.standardize_normal(x, mean=self.distributions.parameters['loc'],
                                                  std=self.distributions.parameters['scale'])

        # evaluate standard Hermite polynomial, orthogonal w.r.t. the PDF of N(0,1)
        h = eval_hermitenorm(self.degree, x_normed)

        # normalization constant
        st_herm_norm = np.sqrt(math.factorial(self.degree))

        h = h / st_herm_norm

        return h