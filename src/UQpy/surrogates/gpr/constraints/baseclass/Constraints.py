from abc import ABC, abstractmethod
import numpy as np


class ConstraintsGPR(ABC):
    """
    Abstract base class of all Constraints. Serves as a template for creating new Kriging constraints for log-likelihood
    function.
    """

    @abstractmethod
    def constraints(self, x_train, y_train, predict_function):
        """
        Abstract method that needs to be implemented by the user when creating a new Correlation function.
        """
        pass