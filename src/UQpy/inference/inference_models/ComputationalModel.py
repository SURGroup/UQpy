from typing import Callable

from beartype import beartype

from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.baseclass import Distribution
from UQpy.inference.inference_models.baseclass.InferenceModel import *
from UQpy.distributions.collection.MultivariateNormal import MultivariateNormal
from UQpy.utilities.ValidationTypes import *
from UQpy.distributions.collection import Normal

import warnings

warnings.filterwarnings('ignore')


class ComputationalModel(InferenceModel):
    @beartype
    def __init__(self, n_parameters: PositiveInteger, runmodel_object: RunModel,
                 error_covariance: Union[np.ndarray, float] = 1.0, name: str = "", prior: Distribution = None,
                 log_likelihood: Callable = None):
        """
        Define a (non-)Gaussian error model for inference.

        :param n_parameters: Number of parameters to be estimated.
        :param runmodel_object: :class:`.RunModel` class object that defines the forward model. This input is required
         for **cases 1a and 1b**.
        :param error_covariance: Covariance for Gaussian error model **(case 1a)**. It can be a scalar (in which case
         the covariance matrix is the identity times that value), a 1D :class:`numpy.ndarray` in which case the
         covariance is assumed to be diagonal, or a full covariance matrix (2D :class:`numpy.ndarray`).
         Default value is :math:`1`.
        :param name: Name of model - optional but useful in a model selection setting.
        :param prior: Prior distribution, must have a :py:meth:`log_pdf` or :py:meth:`pdf` method.
        :param log_likelihood: Function that defines the log-likelihood model, possibly in conjunction with the
         `runmodel_object` **(cases 1b and 2)**. Default is :any:`None`, and a Gaussian-error model is considered
         **(case 1a)**.
        """
        super().__init__(n_parameters, name)
        self.runmodel_object = runmodel_object
        self.error_covariance = error_covariance
        self.log_likelihood = log_likelihood

        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError("UQpy: Input prior should be an object of class Distribution.")
            if not hasattr(self.prior, "log_pdf"):
                if not hasattr(self.prior, "pdf"):
                    raise AttributeError("UQpy: Input prior should have a log_pdf or pdf method.")
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))

    def evaluate_log_likelihood(self, parameters: NumpyFloatArray, data: NumpyFloatArray):
        self.runmodel_object.run(samples=parameters, append_samples=False)
        model_outputs = self.runmodel_object.qoi_list

        # Case 1.a: Gaussian error model
        if self.log_likelihood is None:
            if isinstance(self.error_covariance, (float, int)):
                norm = Normal(loc=0.0, scale=np.sqrt(self.error_covariance))
                log_like_values = np.array(
                    [np.sum([norm.log_pdf(data_i - outpt_i) for data_i, outpt_i in zip(data, output)])
                     for output in model_outputs])
            else:
                multivariate_normal = MultivariateNormal(data, cov=self.error_covariance)
                log_like_values = np.array(
                    [multivariate_normal.log_pdf(x=np.array(output).reshape((-1,))) for output in model_outputs])

        # Case 1.b: likelihood is user-defined
        else:
            log_like_values = self.log_likelihood(data=data, model_outputs=model_outputs, params=parameters)
            if not isinstance(log_like_values, np.ndarray):
                log_like_values = np.array(log_like_values)
            if log_like_values.shape != (parameters.shape[0],):
                raise ValueError(
                    "UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood "
                    "values.")
        return log_like_values
