from abc import ABC, abstractmethod

from UQpy.surrogates.polynomial_chaos.polynomials.PolynomialBasis import PolynomialBasis


class Regression(ABC):

    def __init__(self, polynomial_basis: PolynomialBasis):
        self.polynomial_basis = polynomial_basis

    @abstractmethod
    def run(self, x, y):
        pass
