from abc import ABC, abstractmethod


class InformationCriterion(ABC):

    @abstractmethod
    def evaluate_criterion(self, n_data: int, n_parameters: int) -> float:
        """
        Function that must be implemented by the user in order to create new concrete implementation of the
        :class:`.InformationCriterion` baseclass.

        :param n_data: Number of data points.
        :param n_parameters: Number of parameters characterizing the model.
        :return: The value of the information theoretic criterion
        """
        pass
