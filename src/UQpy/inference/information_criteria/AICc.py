from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion


class AICc(InformationCriterion):
    def evaluate_criterion(self, n_data, n_parameters):
        return 2 * n_parameters + (2 * n_parameters ** 2 + 2 * n_parameters) / (n_data - n_parameters - 1)
