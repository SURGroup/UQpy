from abc import ABC, abstractmethod


class Criterion(ABC):

    def __init(self, ordered_parameters=None, **kwargs):
        self.parameters = kwargs
        self.ordered_parameters = ordered_parameters if not None else tuple(kwargs.keys())
        if len(self.ordered_parameters) != len(self.parameters):
            raise ValueError('Inconsistent dimensions between order_params tuple and params dictionary.')

    @abstractmethod
    def generate_samples(self):
        pass
