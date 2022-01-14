from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion


class AIC(InformationCriterion):
    def evaluate_criterion(self, n_data, n_parameters):
        return 2 * n_parameters
