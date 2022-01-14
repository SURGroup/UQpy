from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion
import numpy as np


class BIC(InformationCriterion):
    def evaluate_criterion(self, n_data, n_parameters):
        return np.log(n_data) * n_parameters
