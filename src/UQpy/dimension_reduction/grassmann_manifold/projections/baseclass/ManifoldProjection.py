from abc import ABC, abstractmethod

from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class ManifoldProjection(ABC):
    """
    The parent class to all classes used to represent data on the Grassmann manifold.
    """