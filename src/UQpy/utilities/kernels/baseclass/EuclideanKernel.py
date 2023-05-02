from abc import ABC

from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class EuclideanKernel(Kernel, ABC):
    """This is a blueprint for Euclidean kernels implemented in the :py:mod:`kernels` module ."""
