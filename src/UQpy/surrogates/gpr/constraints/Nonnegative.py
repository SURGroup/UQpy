from UQpy.surrogates.gpr.constraints.baseclass.Constraints import *


class Nonnegative(ConstraintsGPR):
    def __init__(self, candidate_points, observed_error=0.01, z_value=2):
        self.candidate_points = candidate_points
        self.observed_error = observed_error
        self.z_value = z_value
        self.args = None

    def constraints(self, x_train, y_train, predict_function):
        self.args = (predict_function, self.candidate_points, self.observed_error, self.z_value, x_train, y_train)
        return self.constraints_candidate

    @staticmethod
    def constraints_candidate(theta_, pred, cand_points, obs_err, z_, x_t, y_t):
        tmp_predict, tmp_error = pred(cand_points, True, hyperparameters=10**theta_)
        constraint1 = tmp_predict - z_ * tmp_error

        tmp_predict2 = pred(x_t, False, hyperparameters=10**theta_)
        constraint2 = obs_err - np.abs(y_t[:, 0] - tmp_predict2)

        constraints = np.concatenate((constraint1, constraint2), axis=None)
        return constraints
