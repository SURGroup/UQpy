import logging
from abc import ABC, abstractmethod

import numpy as np

from UQpy.utilities.ValidationTypes import PositiveInteger


class InferenceModel(ABC):

    # Last Modified: 05/13/2020 by Audrey Olivier
    def __init__(
            self,
            n_parameters: PositiveInteger,
            name: str = "",
    ):
        """
        Define a probabilistic model for inference.

        :param n_parameters: Number of parameters to be estimated.
        :param name: Name of model - optional but useful in a model selection setting.
        """
        # Initialize some parameters
        self.prior = None
        self.n_parameters = n_parameters
        self.name = name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def evaluate_log_likelihood(self, parameters: np.ndarray, data: np.ndarray):
        """
        Evaluate the log likelihood, :code:`log p(data|parameters)`.

        This method is the central piece of the :py:mod:`inference` module, it is being called repeatedly by all other
        inference classes to evaluate the likelihood of the data. The log-likelihood can be evaluated at several
        parameter vectors at once, i.e., `parameters` is an :class:`numpy.ndarray` of shape
        :code:`(nsamples, n_parameters)`. If the :class:`.InferenceModel` is powered by :class:`.RunModel` the
        :meth:`RunModel.run` method is called here, possibly  leveraging its parallel execution.

        :param parameters: Parameter vector(s) at which to evaluate the likelihood function, :class:`numpy.ndarray` of
         shape :code:`(nsamples, n_parameters)`.
        :param data: Data from which to learn. For case **1b**, this should be an :class:`numpy.ndarray` of shape
         :code:`(ndata, )`. For **case 3**, it must be an :class:`numpy.ndarray` of shape :code:`(ndata, dimension)`.
         For other cases it must be consistent with the definition of the :meth:`log_likelihood` callable input.
        :return: Log-likelihood evaluated at all `nsamples` parameter vector values, :class:`numpy.ndarray` of shape
         :code:`(nsamples, )`.
        """
        pass

    def evaluate_log_posterior(self, parameters: np.ndarray, data: np.ndarray):
        """
        Evaluate the scaled log posterior :code:`log(p(data|parameters)p(parameters))`.

        This method is called by classes that perform Bayesian inference. If the :class:`.InferenceModel` object does
        not possess a prior, an uninformative prior :code:`p(parameters)=1` is assumed. Warning: This is an improper
        prior.

        :param parameters: Parameter vector(s) at which to evaluate the log-posterior, :class:`numpy.ndarray` of
         shape :code:`(nsamples, n_parameters)`.
        :param data: Data from which to learn. See :py:meth:`evaluate_log_likelihood` method for details.
        :return: Log-posterior evaluated at all `nsamples` parameter vector values, :class:`numpy.ndarray` of shape
         :code:`(nsamples, )`.
        """

        # Compute log likelihood
        log_likelihood_eval = self.evaluate_log_likelihood(parameters=parameters, data=data)

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return log_likelihood_eval

        # Otherwise, use prior provided in the InferenceModel setup
        log_prior_eval = self.prior.log_pdf(x=parameters)

        return log_likelihood_eval + log_prior_eval
