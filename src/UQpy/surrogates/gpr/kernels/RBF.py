from UQpy.surrogates.gpr.kernels.baseclass.Kernel import *


class RBF(Kernel):
    def c(self, x, s, params):
        stack = Kernel.check_samples_and_return_stack(x, s)
        k = params[-1]**2 * np.exp(np.sum(-(1/(2*params[:-1]**2)) * (stack ** 2), axis=2))
        return k
