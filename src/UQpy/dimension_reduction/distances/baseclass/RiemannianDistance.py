from abc import ABC, abstractmethod

from beartype import beartype


class RiemannianDistance(ABC):
    @abstractmethod
    def compute_distance(self, xi, xj) -> float:
        pass

    @staticmethod
    @beartype
    def check_rows(xi, xj):
        if xi.data.shape[0] != xj.data.shape[0]:
            raise ValueError("UQpy: Incompatible dimensions. The matrices must have the same number of rows.")
