from UQpy.surrogates.kriging.correlation_models.baseclass.Correlation import *


class CubicCorrelation(Correlation):
    def c(self, x, s, params, dt=False, dx=False):
        zeta_matrix, dtheta_derivs, dx_derivs = Correlation.derivatives(
            x_=x, s_=s, params=params
        )
        # Initial matrices containing derivates for all values in array. Note since
        # dtheta_s and dx_s already accounted for where derivative should be zero, all
        # that must be done is multiplying the |dij| or thetaj matrix on top of a
        # matrix of derivates w.r.t zeta (in this case, dzeta = -6zeta+6zeta**2)
        drdt = (-6 * zeta_matrix + 6 * zeta_matrix ** 2) * dtheta_derivs
        drdx = (-6 * zeta_matrix + 6 * zeta_matrix ** 2) * dx_derivs
        # Also, create matrix for values of equation, 1 - 3zeta**2 + 2zeta**3, for loop
        zeta_function_cubic = 1 - 3 * zeta_matrix ** 2 + 2 * zeta_matrix ** 3
        rx = np.prod(zeta_function_cubic, 2)
        if dt:
            # Same as previous example, loop over zeta matrix by shifting index
            for i in range(len(params) - 1):
                drdt = drdt * np.roll(zeta_function_cubic, i + 1, axis=2)
            return rx, drdt
        if dx:
            # Same as previous example, loop over zeta matrix by shifting index
            for i in range(len(params) - 1):
                drdx = drdx * np.roll(zeta_function_cubic, i + 1, axis=2)
            return rx, drdx
        return rx
