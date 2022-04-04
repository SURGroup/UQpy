from UQpy.surrogates.kriging.correlation_models.baseclass.Correlation import *


class SplineCorrelation(Correlation):
    def c(self, x, s, params, dt=False, dx=False):
        # x_, s_ = np.atleast_2d(x_), np.atleast_2d(s_)
        # # Create stack matrix, where each block is x_i with all s
        # stack = np.tile(np.swapaxes(np.atleast_3d(x_), 1, 2), (1, np.size(s_, 0), 1)) - np.tile(s_, (
        #     np.size(x_, 0),
        #     1, 1))
        stack = Correlation.check_samples_and_return_stack(x, s)
        # In this case, the zeta value is just abs(stack)*parameters, no comparison
        zeta_matrix = abs(stack) * params
        # So, dtheta and dx are just |dj| and theta*sgn(dj), respectively
        dtheta_derivs = abs(stack)
        # dx_derivs = np.ones((np.size(x,0),np.size(s,0),np.size(s,1)))*parameters
        dx_derivs = np.sign(stack) * params

        # Initialize empty sigma and dsigma matrices
        sigma = np.ones(
            (zeta_matrix.shape[0], zeta_matrix.shape[1], zeta_matrix.shape[2])
        )
        dsigma = np.ones(
            (zeta_matrix.shape[0], zeta_matrix.shape[1], zeta_matrix.shape[2])
        )

        # Loop over cases to create zeta_matrix and subsequent dR matrices
        for i in range(zeta_matrix.shape[0]):
            for j in range(zeta_matrix.shape[1]):
                for k in range(zeta_matrix.shape[2]):
                    y = zeta_matrix[i, j, k]
                    if 0 <= y <= 0.2:
                        sigma[i, j, k] = 1 - 15 * y ** 2 + 30 * y ** 3
                        dsigma[i, j, k] = -30 * y + 90 * y ** 2
                    elif 0.2 < y < 1.0:
                        sigma[i, j, k] = 1.25 * (1 - y) ** 3
                        dsigma[i, j, k] = 3.75 * (1 - y) ** 2 * -1
                    elif y >= 1:
                        sigma[i, j, k] = 0
                        dsigma[i, j, k] = 0

        rx = np.prod(sigma, 2)

        if dt:
            # Initialize derivative matrices incorporating chain rule
            drdt = dsigma * dtheta_derivs
            # Loop over to create proper matrices
            for i in range(len(params) - 1):
                drdt = drdt * np.roll(sigma, i + 1, axis=2)
            return rx, drdt
        if dx:
            # Initialize derivative matrices incorporating chain rule
            drdx = dsigma * dx_derivs
            # Loop over to create proper matrices
            for i in range(len(params) - 1):
                drdx = drdx * np.roll(sigma, i + 1, axis=2)
            return rx, drdx
        return rx
