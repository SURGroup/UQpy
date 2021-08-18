from abc import ABC, abstractmethod


class Regression(ABC):
    @abstractmethod
    def run(self, x, y):
        pass