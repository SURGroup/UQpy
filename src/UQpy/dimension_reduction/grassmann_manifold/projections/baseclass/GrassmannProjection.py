from abc import ABC, abstractmethod

from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class GrassmannProjection(ABC):
    """
    The parent class to all classes used to project data onto the Grassmann manifold.
    """