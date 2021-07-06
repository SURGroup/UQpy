from abc import ABC, abstractmethod


class Regression(ABC):

    @abstractmethod
    def r(self, s):
        pass
