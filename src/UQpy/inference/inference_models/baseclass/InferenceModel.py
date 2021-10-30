import logging

import numpy as np
from abc import ABC, abstractmethod
from UQpy.distributions import Distribution
from UQpy.RunModel import RunModel
from UQpy.utilities.ValidationTypes import PositiveInteger


class InferenceModel(ABC):

    # Last Modified: 05/13/2020 by Audrey Olivier
    def __init__(
        self,
        parameters_number: PositiveInteger,
        runmodel_object: RunModel = None,
        log_likelihood=None,
        distributions=None,
        name: str = "",
        error_covariance: float = 1.0,
        prior: Distribution = None,
    ):
        """
        Define a probabilistic model for inference.

        :param parameters_number: Number of parameters to be estimated.
        :param runmodel_object: ``RunModel`` class object that defines the forward model. This input is required for
         cases 1a and 1b.
        :param log_likelihood: Function that defines the log-likelihood model, possibly in conjunction with the
         `runmodel_object` (cases 1b and 2). Default is None, and a Gaussian-error model is considered (case 1a).
        |  If a `runmodel_object` is also defined (case 1b), this function is called as:
        |  `model_outputs = runmodel_object.run(samples=params).qoi_list`
        |  `log_likelihood(params, model_outputs, data, **kwargs_likelihood)`
        |  If no `runmodel_object` is defined (case 2), this function is called as:
        |  `log_likelihood(params, data, **kwargs_likelihood)`
        :param distributions: Distribution :math:`\pi` for which to learn parameters from iid data (case 3).
         When creating this ``Distribution`` object, the parameters to be learned should be set to `None`.
        :param name: Name of model - optional but useful in a model selection setting.
        :param error_covariance: Covariance for Gaussian error model (case 1a). It can be a scalar (in which case the
         covariance matrix is the identity times that value), a 1d `ndarray` in which case the covariance is assumed to
         be diagonal, or a full covariance matrix (2D `ndarray`). Default value is 1.
        :param prior: Prior distribution, must have a `log_pdf` or `pdf` method.
        """
        # Initialize some parameters
        self.parameters_number = parameters_number
        self.name = name
        self.logger = logging.getLogger(__name__)

        self.runmodel_object = runmodel_object
        self.error_covariance = error_covariance
        self.log_likelihood = log_likelihood
        self.distributions = distributions
        # Perform checks on inputs runmodel_object, log_likelihood, distribution_object that define the inference model
        if (
            (self.runmodel_object is None)
            and (self.log_likelihood is None)
            and (self.distributions is None)
        ):
            raise ValueError(
                "UQpy: One of runmodel_object, log_likelihood or dist_object inputs must be provided."
            )
        if self.runmodel_object is not None and (
            not isinstance(self.runmodel_object, RunModel)
        ):
            raise TypeError(
                "UQpy: Input runmodel_object should be an object of class RunModel."
            )
        if (self.log_likelihood is not None) and (not callable(self.log_likelihood)):
            raise TypeError("UQpy: Input log_likelihood should be a callable.")
        if self.distributions is not None:
            if (self.runmodel_object is not None) or (self.log_likelihood is not None):
                raise ValueError(
                    "UQpy: Input dist_object cannot be provided concurrently with log_likelihood "
                    "or runmodel_object."
                )
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
            # Check which parameters need to be updated (i.e., those set as None)
            init_params = self.distributions.get_parameters()
            self.list_params = [
                key
                for key in self.distributions.ordered_parameters
                if init_params[key] is None
            ]
            if len(self.list_params) != self.parameters_number:
                raise TypeError(
                    "UQpy: Incorrect dimensions between parameters_number and number"
                    " of inputs set to None."
                )

        # Define prior if it is given
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

    @abstractmethod
    def evaluate_log_likelihood(self, params, data):
        """
        Evaluate the log likelihood, `log p(data|params)`.

        This method is the central piece of the ``inference`` module, it is being called repeatedly by all other
        ``inference`` classes to evaluate the likelihood of the data. The log-likelihood can be evaluated at several
        parameter vectors at once, i.e., `params` is an `ndarray` of shape (nsamples, nparams). If the
        ``InferenceModel`` is powered by ``RunModel`` the ``RunModel.run`` method is called here, possibly leveraging
        its parallel execution.

        :param params: Parameter vector(s) at which to evaluate the likelihood function, `ndarray` of shape
         `(nsamples, nparams)`.
        :param data: Data from which to learn. For case 1b, this should be an `ndarray` of shape `(ndata, )`. For case
         3, it must be an `ndarray` of shape `(ndata, dimension)`. For other cases it must be consistent with the
         definition of the ``log_likelihood`` callable input.
         :return Log-likelihood evaluated at all `nsamples` parameter vector values, `ndarray` of shape (nsamples, ).
        """
        pass

    def evaluate_log_posterior(self, parameter_vector, data):
        """
        Evaluate the scaled log posterior `log(p(data|params)p(params))`.

        This method is called by classes that perform Bayesian inference. If the ``InferenceModel`` object does not
        possess a prior, an uninformative prior `p(params)=1` is assumed. Warning: This is an improper prior.

        :param parameter_vector: Parameter vector(s) at which to evaluate the log-posterior, `ndarray` of shape
         (nsamples, nparams).
        :param data: Data from which to learn. See `evaluate_log_likelihood` method for details.
        :return: Log-posterior evaluated at all `nsamples` parameter vector values, `ndarray` of shape (nsamples, ).
        """

        # Compute log likelihood
        log_likelihood_eval = self.evaluate_log_likelihood(
            params=parameter_vector, data=data
        )

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return log_likelihood_eval

        # Otherwise, use prior provided in the InferenceModel setup
        log_prior_eval = self.prior.log_pdf(x=parameter_vector)

        return log_likelihood_eval + log_prior_eval
