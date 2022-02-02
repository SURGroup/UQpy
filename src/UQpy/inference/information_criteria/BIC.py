from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion
import numpy as np
from UQpy.inference import MLE
from UQpy.utilities.ValidationTypes import NumpyFloatArray


class BIC(InformationCriterion):

    def minimize_criterion(self,
                           data: NumpyFloatArray,
                           parameter_estimator: MLE,
                           return_penalty: bool = False):
        inference_model = parameter_estimator.inference_model
        max_log_like = parameter_estimator.max_log_like
        n_parameters = inference_model.n_parameters
        n_data = len(data)

        penalty_term = self._calculate_penalty_term(n_data, n_parameters)
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like + penalty_term

    def _calculate_penalty_term(self, n_data, n_parameters):
        return np.log(n_data) * n_parameters
