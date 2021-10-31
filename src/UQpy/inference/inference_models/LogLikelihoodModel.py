from beartype import beartype

from UQpy.inference.inference_models.baseclass.InferenceModel import *


class LogLikelihoodModel(InferenceModel):
    @beartype
    def __init__(
        self, parameters_number: PositiveInteger, log_likelihood, name: str = ""
    ):
        """
        Define a log-likelihood model for inference.

        :param parameters_number: Number of parameters to be estimated.
        :param log_likelihood: Function that defines the log-likelihood model, possibly in conjunction with the
         `runmodel_object` (cases 1b and 2). Default is None, and a Gaussian-error model is considered (case 1a).
         |  If a `runmodel_object` is also defined (case 1b), this function is called as:
         |  `model_outputs = runmodel_object.run(samples=params).qoi_list`
         |  `log_likelihood(params, model_outputs, data, **kwargs_likelihood)`
         |  If no `runmodel_object` is defined (case 2), this function is called as:
         |  `log_likelihood(params, data, **kwargs_likelihood)`
        :param name: Name of model - optional but useful in a model selection setting.
        """
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
