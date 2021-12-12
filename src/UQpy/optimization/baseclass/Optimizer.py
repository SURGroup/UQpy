from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, bounds):
        self._bounds = bounds

    @abstractmethod
    def optimize(self, function, initial_guess, args, jac):
        pass

    @abstractmethod
    def apply_constraints(self, constraints):
        pass

    @abstractmethod
    def update_bounds(self, bounds):
        pass

    @abstractmethod
    def supports_jacobian(self):
        pass

    @property
    def bounds(self):
        return self._bounds
