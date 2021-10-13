from beartype import beartype

from UQpy.inference.inference_models.baseclass.InferenceModel import *


class LogLikelihoodModel(InferenceModel):
    @beartype
    def __init__(
        self, parameters_number: PositiveInteger, log_likelihood, name: str = ""
    ):

        self.name = name
        self.log_likelihood = log_likelihood
        self.parameters_number = parameters_number

    def evaluate_log_likelihood(self, params, data):
        log_like_values = self.log_likelihood(data=data, params=params)
        if not isinstance(log_like_values, np.ndarray):
            log_like_values = np.array(log_like_values)
        if log_like_values.shape != (params.shape[0],):
            raise ValueError(
                "UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood values."
            )
        return log_like_values
