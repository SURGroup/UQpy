from abc import ABC, abstractmethod

from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import PolynomialBasis


class Regression(ABC):

    @abstractmethod
    def run(self, x, y, polynomial_basis):
        pass
