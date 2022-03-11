from typing import Callable

import numpy as np
from beartype import beartype
import warnings

warnings.filterwarnings('ignore')

from UQpy.inference.inference_models.baseclass.InferenceModel import *


class LogLikelihoodModel(InferenceModel):
    @beartype
    def __init__(self, n_parameters: PositiveInteger, log_likelihood: Callable, name: str = ""):
        """
        Define a log-likelihood model for inference.

        :param n_parameters: Number of parameters to be estimated.
        :param log_likelihood: Function that defines the log-likelihood model.
        :param name: Name of model - optional but useful in a model selection setting.
        """
        super().__init__(n_parameters, name)
        self.name = name
        self.log_likelihood = log_likelihood
        self.n_parameters = n_parameters

    def evaluate_log_likelihood(self, parameters: np.ndarray, data: np.ndarray):
        log_like_values = self.log_likelihood(data=data, params=parameters)
        if not isinstance(log_like_values, np.ndarray):
            log_like_values = np.array(log_like_values)
        if log_like_values.shape != (parameters.shape[0],):
            raise ValueError("UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood values.")
        return log_like_values
