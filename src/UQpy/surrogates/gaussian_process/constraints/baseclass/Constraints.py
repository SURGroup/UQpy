from abc import ABC, abstractmethod


class ConstraintsGPR(ABC):
    """
    Abstract base class of all Constraints. Serves as a template for creating new Kriging constraints for log-likelihood
    function.
    """

    @abstractmethod
    def define_arguments(self, x_train, y_train, predict_function):
        """
        Abstract method that needs to be implemented by the user which stores all the arguments in a dictionary and
        return that dictionary inside a list.
        """
        pass


    @staticmethod
    def constraints(theta_, kwargs):
        """
        A static method, which take hyperaparameters and constraints argument and evaluate constraints value.
        """
        pass
