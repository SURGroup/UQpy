from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.inference.MLE import MLE


class InformationCriterion(ABC):

    @abstractmethod
    def minimize_criterion(self, data: np.ndarray,
                           parameter_estimator: Union[MLE, BayesParameterEstimation],
                           return_penalty: bool = False) -> float:
        """
        Function that must be implemented by the user in order to create new concrete implementation of the
        :class:`.InformationCriterion` baseclass.
        """
        pass
