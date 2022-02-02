from abc import ABC, abstractmethod


class InformationCriterion(ABC):

    @abstractmethod
    def minimize_criterion(self, data,
                           parameter_estimator,
                           return_penalty=False) -> float:
        """
        Function that must be implemented by the user in order to create new concrete implementation of the
        :class:`.InformationCriterion` baseclass.
        """
        pass
