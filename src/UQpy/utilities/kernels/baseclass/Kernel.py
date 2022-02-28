import itertools
from abc import ABC, abstractmethod
import numpy as np

from UQpy.utilities.GrassmannPoint import GrassmannPoint


class Kernel(ABC):

    @abstractmethod
    def kernel_entry(self, xi, xj):
        pass
