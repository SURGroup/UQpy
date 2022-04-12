from UQpy.surrogates.gaussian_process.kernels.baseclass.Kernel import *


class RBF(Kernel):
    def c(self, x, s, params):
        """
        This method compute the RBF kernel on sample points 'x' and 's'.

        :params x: An array containing input samples.
        :params s: An array containing input samples.
        :params params: A list/array of hyperparameters containing length scales and the process variance.
        """
        stack = Kernel.check_samples_and_return_stack(x/params[:-1], s/params[:-1])
        k = params[-1] ** 2 * np.exp(np.sum(-0.5 * (stack ** 2), axis=2))
        return k
