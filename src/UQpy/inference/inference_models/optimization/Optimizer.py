from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, function, initial_guess):
        pass
