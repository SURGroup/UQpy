from UQpy.surrogates.kriging.correlation_models.baseclass.Correlation import *


class GaussianCorrelation(Correlation):
    def c(self, x, s, params, dt=False, dx=False):
        stack = Correlation.check_samples_and_return_stack(x, s)
        rx = np.exp(np.sum(-params * (stack ** 2), axis=2))
        if dt:
            drdt = -(stack ** 2) * np.transpose(
                np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0)
            )
            return rx, drdt
        if dx:
            drdx = (
                -2
                * params
                * stack
                * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
            )
            return rx, drdx
        return rx
