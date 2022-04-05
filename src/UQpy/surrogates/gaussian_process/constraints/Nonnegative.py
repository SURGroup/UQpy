from UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints import *
import numpy as np


class Nonnegative(ConstraintsGPR):
    def __init__(self, constraint_points, observed_error=0.01, z_value=2):
        """
        Nonnegative class defines constraints for the MLE optimization problem, such that `GaussianProcessRegressor`
        surrogate prediction has positive prediction over the input domain.

        :params constraint_points: Points over which 'Z' standard deviation below the mean is still nonnegative.
        :params z_value: The value 'Z' used in the inequality for constraint points.
         Default: z_value = 2.
        :params observed_error: Tolerance on the absolute difference between observed output and its prediction.
        """
        self.constraint_points = constraint_points
        self.observed_error = observed_error
        self.z_value = z_value
        self.args = None

    def define_arguments(self, x_train, y_train, predict_function):
        """
        :params x_train: Input training data, used to train the GPR.
        :params y_train: Output training data.
        :params prediction_function: The 'predict' method from the GaussianProcessRegressor
        """
        self.args = (predict_function, self.constraint_points, self.observed_error, self.z_value, x_train, y_train)
        return self.constraints

    @staticmethod
    def constraints(theta_, pred, const_points, obs_err, z_, x_t, y_t):
        tmp_predict, tmp_error = pred(const_points, True, hyperparameters=10**theta_)
        constraint1 = tmp_predict - z_ * tmp_error

        tmp_predict2 = pred(x_t, False, hyperparameters=10**theta_)
        constraint2 = obs_err - np.abs(y_t[:, 0] - tmp_predict2)

        constraints = np.concatenate((constraint1, constraint2), axis=None)
        return constraints
