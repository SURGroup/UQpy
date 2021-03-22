import scipy.special as special

from UQpy.Distributions import Uniform
from UQpy.Surrogates.PCE.Polynomials import Polynomials


class Legendre(Polynomials):
    """
    Class of univariate polynomials appropriate for data generated from a
    uniform distribution.

    **Inputs:**

    * **degree** ('int'):
        Maximum degree of the polynomials.

    * **dist_object** ('class'):
        Distribution object of the generated samples.

    **Methods:**
    """

    def __init__(self, degree, dist_object):
        super().__init__(dist_object, degree)
        self.degree = degree
        self.pdf = self.dist_object.pdf

    def get_polys(self, x):
        """
        Calculates the normalized Legendre polynomials evaluated at sample points.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the samples.

        * **y** (`ndarray`):
            `ndarray` containing the samples.

        **Outputs:**

        (`list`):
            Returns a list of 'ndarrays' with the design matrix and the
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