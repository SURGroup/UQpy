from abc import ABC, abstractmethod


class LearningFunction(ABC):
    def __init(self, ordered_parameters=None, **kwargs):
        self.parameters = kwargs
        self.ordered_parameters = (ordered_parameters if ordered_parameters is not None else tuple(kwargs.keys()))
        if len(self.ordered_parameters) != len(self.parameters):
            raise ValueError("Inconsistent dimensions between order_params tuple and params dictionary.")

    @abstractmethod
    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
        """
        Abstract method that needs to be overriden by the user to create new Adaptive Kriging Learning functions.
        """
        pass
