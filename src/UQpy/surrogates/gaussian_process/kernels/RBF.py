from typing import Union

from UQpy.surrogates.gaussian_process.kernels.baseclass.Kernel import *


class RBF(Kernel):
    def __init__(self, kernel_parameter: Union[int, float] = 1.0):
        super().__init__(kernel_parameter)

    def calculate_kernel_matrix(self, x, s):
        """
        This method compute the RBF kernel on sample points 'x' and 's'.

        :params x: An array containing training points.
        :params s: An array containing input points.
        """
        stack = Kernel.check_samples_and_return_stack(x / self.kernel_parameter, s / self.kernel_parameter)
        return np.exp(np.sum(-0.5 * (stack ** 2), axis=2))
