from UQpy.surrogates.kriging.correlation_models.baseclass.Correlation import *


class LinearCorrelation(Correlation):
    def c(self, x, s, params, dt=False, dx=False):
        stack = Correlation.check_samples_and_return_stack(x, s)
        # Taking stack and turning each d value into 1-theta*dij
        after_parameters = 1 - params * abs(stack)
        # Define matrix of zeros to compare against (not necessary to be defined separately,
        # but the line is bulky if this isn't defined first, and it is used more than once)
        comp_zero = np.zeros((np.size(x, 0), np.size(s, 0), np.size(s, 1)))
        # Compute matrix of max{0,1-theta*d}
        max_matrix = np.maximum(after_parameters, comp_zero)
        rx = np.prod(max_matrix, 2)
        # Create matrix that has 1s where max_matrix is nonzero
        # -Essentially, this acts as a way to store the indices of where the values are nonzero
        ones_and_zeros = max_matrix.astype(bool).astype(int)
        # Set initial derivatives as if all were positive
        first_dtheta = -abs(stack)
        first_dx = np.negative(params) * np.sign(stack)
        # Multiply derivs by ones_and_zeros...this will set the values where the
        # derivative should be zero to zero, and keep all other values the same
        drdt = np.multiply(first_dtheta, ones_and_zeros)
        drdx = np.multiply(first_dx, ones_and_zeros)
        if dt:
            # Loop over parameters, shifting max_matrix and multiplying over derivative matrix with each iter
            for i in range(len(params) - 1):
                drdt = drdt * np.roll(max_matrix, i + 1, axis=2)
            return rx, drdt
        if dx:
            # Loop over parameters, shifting max_matrix and multiplying over derivative matrix with each iter
            for i in range(len(params) - 1):
                drdx = drdx * np.roll(max_matrix, i + 1, axis=2)
            return rx, drdx
        return rx
