from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Kernel(ABC):
    """
    Abstract base class of all Kernels. Serves as a template for creating new Gaussian Process covariance
    functions.
    """

    def __init__(self, kernel_parameter: Union[int, float]):
        self.__kernel_parameter = kernel_parameter
        self.kernel_matrix=None

    @property
    def kernel_parameter(self):
        return self.__kernel_parameter

    @kernel_parameter.setter
    def kernel_parameter(self, value):
        self.__kernel_parameter = value


    @abstractmethod
    def calculate_kernel_matrix(self, x, s):
        """
        Abstract method that needs to be implemented by the user when creating a new Covariance function.
        """
        pass

    @staticmethod
    def check_samples_and_return_stack(x, s):
        x_, s_ = np.atleast_2d(x), np.atleast_2d(s)
        # Create stack matrix, where each block is x_i with all s
        stack = np.tile(
            np.swapaxes(np.atleast_3d(x_), 1, 2), (1, np.size(s_, 0), 1)
        ) - np.tile(s_, (np.size(x_, 0), 1, 1))
        return stack
