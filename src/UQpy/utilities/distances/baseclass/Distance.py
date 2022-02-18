from abc import abstractmethod, ABC


class Distance(ABC):

    @abstractmethod
    def compute_distance(self, xi, xj) -> float:
        pass
