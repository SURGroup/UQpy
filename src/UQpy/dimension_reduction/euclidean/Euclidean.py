import itertools
from UQpy.dimension_reduction.kernels.euclidean.Gaussian import Gaussian
import numpy as np
import scipy.spatial.distance as sd


class Euclidean:

    def __init__(self, data, epsilon=None):
        self.data = data

    def evaluate_kernel_matrix(self, kernel=Gaussian()):
        kernel_matrix = kernel.apply_method(self.data)
        return kernel_matrix
