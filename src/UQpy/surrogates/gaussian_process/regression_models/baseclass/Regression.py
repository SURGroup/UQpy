from abc import ABC, abstractmethod


class Regression(ABC):
    """
    Abstract base class of all Regressions. Serves as a template for creating new Gaussian Process regression
    functions.
    """
    @abstractmethod
    def r(self, s):
        """
        Abstract method that needs to be implemented by the user when creating a new Regression function.
        """
        pass
