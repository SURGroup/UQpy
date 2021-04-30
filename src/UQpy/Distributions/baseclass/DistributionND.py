import numpy as np
from .Distribution import Distribution

########################################################################################################################
#        Multivariate Continuous Distributions
########################################################################################################################


class DistributionND(Distribution):
    """
    Parent class for multivariate probability distributions.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _check_x_dimension(x, d=None):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints, d)
        """
        x = np.array(x)
        if len(x.shape) != 2:
            raise ValueError("Wrong dimension in x.")
        if (d is not None) and (x.shape[1] != d):
            raise ValueError("Wrong dimension in x.")
        return x
