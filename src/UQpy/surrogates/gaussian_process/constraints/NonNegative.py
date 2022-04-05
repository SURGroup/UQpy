from UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints import *
import numpy as np


class NonNegative(ConstraintsGPR):
    def __init__(self, constraint_points, observed_error=0.01, z_value=2):
        """
        Nonnegative class defines constraints for the MLE optimization problem, such that `GaussianProcessRegressor`
        surrogate prediction has positive prediction over the input domain.

        :params constraint_points: Points over which 'Z' standard deviation below the mean is still nonnegative.
        :params z_value: The value 'Z' used in the inequality for constraint points.
         Default: z_value = 2.
        :params observed_error: Error tolerance between observed output and its prediction.
        """
        self.constraint_points = constraint_points
        self.observed_error = observed_error
        self.z_value = z_value
        self.kwargs = {}
        self.constraint_args = None

    def define_arguments(self, x_train, y_train, predict_function):
        """
        :params x_train: Input training data, used to train the GPR.
        :params y_train: Output training data.
        :params prediction_function: The 'predict' method from the GaussianProcessRegressor
        """
        self.kwargs['x_t'] = x_train
        self.kwargs['y_t'] = y_train
        self.kwargs['pred'] = predict_function
        self.kwargs['const_points'] = self.constraint_points
        self.kwargs['obs_err'] = self.observed_error
        self.kwargs['z_'] = self.z_value
        self.constraint_args = [self.kwargs]
        return self.constraints

    @staticmethod
    def constraints(theta_, kwargs):
        """
        :param theta_: Log-transformed hyperparameters.
        :params kwargs: A dictionary with all arguments as defined in `define_arguments` method.
        """
        x_t, y_t, pred = kwargs['x_t'], kwargs['y_t'], kwargs['pred']
        const_points, obs_err, z_ = kwargs['const_points'], kwargs['obs_err'], kwargs['z_']
        tmp_predict, tmp_error = pred(const_points, True, hyperparameters=10**theta_)
        constraint1 = tmp_predict - z_ * tmp_error

        tmp_predict2 = pred(x_t, False, hyperparameters=10**theta_)
        constraint2 = obs_err - np.abs(y_t[:, 0] - tmp_predict2)

        constraints = np.concatenate((constraint1, constraint2), axis=None)
        return constraints
