from typing import Union

import numpy as np
from beartype import beartype

from UQpy.distributions.baseclass import Distribution
from UQpy.inference.inference_models.baseclass.InferenceModel import *


class DistributionModel(InferenceModel):
    @beartype
    def __init__(self, distributions: Union[Distribution, list[Distribution]],
                 n_parameters: PositiveInteger, name: str = "", prior: Distribution = None):
        """
        Define a probability distribution model for inference.

        :param distributions: Distribution :math:`\pi` for which to learn parameters from iid data **(case 3)**.
         When creating this :class:`.Distribution` object, the parameters to be learned should be set to :any:`None`.
         Any parameter that is assigned a value will not be learned.
        :param n_parameters: Number of parameters to be estimated.
        :param name: Name of model - optional but useful in a model selection setting.
        :param prior: Prior distribution, must have a :py:meth:`log_pdf` or :py:meth:`pdf` method.
        """
        super().__init__(n_parameters, name)
        self.distributions = distributions

        if self.distributions is not None:
            if not isinstance(self.distributions, Distribution):
                raise TypeError("UQpy: Input dist_object should be an object of class Distribution.")
            if not hasattr(self.distributions, "log_pdf"):
                if not hasattr(self.distributions, "pdf"):
                    raise AttributeError("UQpy: dist_object should have a log_pdf or pdf method.")
                self.distributions.log_pdf = lambda x: np.log(self.distributions.pdf(x))
            init_params = self.distributions.get_parameters()
            self.list_params = [
                key for key in self.distributions.ordered_parameters if init_params[key] is None]
            if len(self.list_params) != self.n_parameters:
                raise TypeError("UQpy: Incorrect dimensions between nparams and number of inputs set to None.")

        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError("UQpy: Input prior should be an object of class Distribution.")
            if not hasattr(self.prior, "log_pdf"):
                if not hasattr(self.prior, "pdf"):
                    raise AttributeError("UQpy: Input prior should have a log_pdf or pdf method.")
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))

    def evaluate_log_likelihood(self, parameters, data):
        log_like_values = []
        for params_ in parameters:
            self.distributions.update_parameters(**dict(zip(self.list_params, params_)))
            log_like_values.append(np.sum(self.distributions.log_pdf(x=data)))
        log_like_values = np.array(log_like_values)
        return log_like_values
