from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    """
    Abstract base class of all Kernels. Serves as a template for creating new Gaussian Process covariance
    functions.
    """

    @abstractmethod
    def c(self, x, s, params):
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

    # @staticmethod
    # def derivatives(x_, s_, params):
    #     stack = Kernel.check_samples_and_return_stack(x_, s_)
    #     # Taking stack and creating array of all thetaj*dij
    #     after_parameters = params * abs(stack)
    #     # Create matrix of all ones to compare
    #     comp_ones = np.ones((np.size(x_, 0), np.size(s_, 0), np.size(s_, 1)))
    #     # zeta_matrix has all values min{1,theta*dij}
    #     zeta_matrix_ = np.minimum(after_parameters, comp_ones)
    #     # Copy zeta_matrix to another matrix that will used to find where derivative should be zero
    #     indices = zeta_matrix_.copy()
    #     # If value of min{1,theta*dij} is 1, the derivative should be 0.
    #     # So, replace all values of 1 with 0, then perform the .astype(bool).astype(int)
    #     # operation like in the linear example, so you end up with an array of 1's where
    #     # the derivative should be caluclated and 0 where it should be zero
    #     indices[indices == 1] = 0
    #     # Create matrix of all |dij| (where non zero) to be used in calculation of dR/dtheta
    #     dtheta_derivs_ = indices.astype(bool).astype(int) * abs(stack)
    #     # Same as above, but for matrix of all thetaj where non-zero
    #     dx_derivs_ = indices.astype(bool).astype(int) * params * np.sign(stack)
    #     return zeta_matrix_, dtheta_derivs_, dx_derivs_
