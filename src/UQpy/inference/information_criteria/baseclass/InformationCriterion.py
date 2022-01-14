from abc import ABC, abstractmethod


class InformationCriterion(ABC):

    @abstractmethod
    def evaluate_criterion(self, n_data, n_parameters):
        pass
