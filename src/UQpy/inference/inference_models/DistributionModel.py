from typing import Union

from beartype import beartype

from UQpy.inference.inference_models.baseclass.InferenceModel import *


class DistributionModel(InferenceModel):
    @beartype
    def __init__(
        self,
        distributions: Union[Distribution, list[Distribution]],
        parameters_number: PositiveInteger,
        name: str = "",
        prior: Distribution = None,
    ):
        """
        Define a distributions model for inference.

        :param distributions: Distribution :math:`\pi` for which to learn parameters from iid data **(case 3)**.
         When creating this :class:`.Distribution` object, the parameters to be learned should be set to `None`.
        :param parameters_number: Number of parameters to be estimated.
        :param name: Name of model - optional but useful in a model selection setting.
        :param prior: Prior distribution, must have a `log_pdf` or `pdf` method.
        """
        self.distributions = distributions
        self.parameters_number = parameters_number
        self.name = name

        if self.distributions is not None:
            if not isinstance(self.distributions, Distribution):
                raise TypeError(
                    "UQpy: Input dist_object should be an object of class Distribution."
                )
            if not hasattr(self.distributions, "log_pdf"):
                if not hasattr(self.distributions, "pdf"):
                    raise AttributeError(
                        "UQpy: dist_object should have a log_pdf or pdf method."
                    )
                self.distributions.log_pdf = lambda x: np.log(self.distributions.pdf(x))
            init_params = self.distributions.get_parameters()
            self.list_params = [
                key
                for key in self.distributions.ordered_parameters
                if init_params[key] is None
            ]
            if len(self.list_params) != self.parameters_number:
                raise TypeError(
                    "UQpy: Incorrect dimensions between nparams and number of inputs set to None."
                )

        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError(
                    "UQpy: Input prior should be an object of class Distribution."
                )
            if not hasattr(self.prior, "log_pdf"):
                if not hasattr(self.prior, "pdf"):
                    raise AttributeError(
                        "UQpy: Input prior should have a log_pdf or pdf method."
                    )
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))

    def evaluate_log_likelihood(self, params, data):
        log_like_values = []
        for params_ in params:
            self.distributions.update_parameters(**dict(zip(self.list_params, params_)))
            log_like_values.append(np.sum(self.distributions.log_pdf(x=data)))
        log_like_values = np.array(log_like_values)
        return log_like_values
