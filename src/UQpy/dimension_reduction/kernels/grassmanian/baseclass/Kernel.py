from abc import ABC, abstractmethod


class Kernel(ABC):

    @abstractmethod
    def apply_method(self, point1, point2):
        pass
