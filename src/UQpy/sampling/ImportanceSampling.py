import logging
from typing import Union, Callable

from beartype import beartype

from UQpy.utilities.ValidationTypes import PositiveInteger, RandomStateType, NumpyFloatArray
from UQpy.utilities.Utilities import process_random_state
from UQpy.distributions import Distribution
import numpy as np


class ImportanceSampling:

    # Last Modified: 10/05/2020 by Audrey Olivier
    @beartype
    def __init__(self,
                 pdf_target: Callable = None,
                 log_pdf_target: Callable = None,
                 args_target: tuple = None,
                 proposal: Union[None, Distribution] = None,
                 random_state: RandomStateType = None,
                 nsamples: PositiveInteger = None):
        """
        Sample from a user-defined target density using importance sampling.

        :param pdf_target: Callable that evaluates the pdf of the target distribution. Either `log_pdf_target` or
         `pdf_target` must be specified (the former is preferred).
        :param log_pdf_target: Callable that evaluates the log-pdf of the target distribution. Either `log_pdf_target`
         or `pdf_target` must be specified (the former is preferred).
        :param args_target: Positional arguments of the target log_pdf / pdf callable.
        :param proposal: Proposal to sample from. This :class:`.Distribution` object must have an :py:meth:`rvs` method
         and a `log_pdf` (or pdf) method.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         :any:`None`.

         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        :param nsamples: Number of samples to generate - see :meth:`run` method. If not :any:`None`, the :py:meth:`run`
         method is called when the object is created. Default is :any:`None`.
        """
        # Initialize proposal: it should have an rvs and log pdf or pdf method
        self.evaluate_log_target = None
        self.proposal = proposal
        self._args_target = args_target
        self.log_pdf_target = log_pdf_target
        self.pdf_target = pdf_target

        self.logger = logging.getLogger(__name__)
        self.random_state = process_random_state(random_state)

        # Initialize the samples and weights
        self.samples: NumpyFloatArray = None
        """Set of samples, :class:`numpy.ndarray` of shape :code:`(nsamples, dimensions)`"""
        self.unnormalized_log_weights: NumpyFloatArray = None
        """Unnormalized log weights, i.e., :code:`log_w(x) = log_target(x) - log_proposal(x)`, :class:`numpy.ndarray` of 
        shape :code:`(nsamples, )`"""
        self.weights: NumpyFloatArray = None
        """Importance weights, weighted so that they sum up to 1, :class:`numpy.ndarray` of shape :code:`(nsamples, )`
        """
        self.unweighted_samples: NumpyFloatArray = None
        """Set of un-weighted samples (useful for instance for plotting), computed by calling the :meth:`resample` 
        method"""

        # Run IS if nsamples is provided
        if nsamples is not None and nsamples != 0:
            self.run(nsamples)

    def _preprocess_proposal(self):
        if not isinstance(self.proposal, Distribution):
            raise TypeError("UQpy: The proposal should be of type Distribution.")
        if not hasattr(self.proposal, "rvs"):
            raise AttributeError("UQpy: The proposal should have an rvs method")
        if not hasattr(self.proposal, "log_pdf"):
            if not hasattr(self.proposal, "pdf"):
                raise AttributeError("UQpy: The proposal should have a log_pdf or pdf method")
            self.proposal.log_pdf = lambda x: np.log(
                np.maximum(self.proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))))

    @beartype
    def run(self, nsamples: PositiveInteger):
        """
        Generate and weight samples.

        This function samples from the proposal and appends samples to existing ones (if any). It then weights the
        samples as :code:`log_w_unnormalized) = log(target)-log(proposal)`.

        :param nsamples: Number of weighted samples to generate.

        This function has no returns, but it updates the output attributes :py:attr:`samples`,
        :py:attr:`unnormalized_log_weights` and :py:attr:`weights` of the :class:`.ImportanceSampling` object.
        """
        if self.evaluate_log_target is None:
            self._preprocess_proposal()
            self.evaluate_log_target = self._preprocess_target(
                log_pdf_=self.log_pdf_target,
                pdf_=self.pdf_target,
                args=self._args_target, )

        self.logger.info("UQpy: Running Importance Sampling...")
        # Sample from proposal
        new_samples = self.proposal.rvs(nsamples=nsamples, random_state=self.random_state)
        # Compute un-scaled weights of new samples
        a = self.evaluate_log_target(x=new_samples)
        new_log_weights = self.evaluate_log_target(x=new_samples) - self.proposal.log_pdf(x=new_samples)

        # Save samples and weights (append to existing if necessary)
        if self.samples is None:
            self.samples = new_samples
            self.unnormalized_log_weights = new_log_weights
        else:
            self.samples = np.concatenate([self.samples, new_samples], axis=0)
            self.unnormalized_log_weights = np.concatenate(
                [self.unnormalized_log_weights, new_log_weights], axis=0)

        # Take the exponential and normalize the weights
        weights = np.exp(self.unnormalized_log_weights - max(self.unnormalized_log_weights))
        # note: scaling with max avoids having NaN of Inf when taking the exp
        sum_w = np.sum(weights, axis=0)
        self.weights = weights / sum_w
        self.logger.info("UQpy: Importance Sampling performed successfully")

        # If a set of unweighted samples exist, delete them as they are not representative of the distribution anymore
        if self.unweighted_samples is not None:
            self.logger.info("UQpy: unweighted samples are being deleted, call the resample method to regenerate them")
            self.unweighted_samples = None

    def resample(self, method: str = "multinomial", nsamples: int = None):
        """
        Resample to get a set of un-weighted samples that represent the target pdf.

        Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
        plotting for instance.

        The :meth:`resample` method is not called automatically when instantiating the :class:`.ImportanceSampling`
        class or when invoking its :meth:`run` method.

        :param method: Resampling method, as of V4 only multinomial resampling is supported. Default: 'multinomial'.
        :param nsamples: Number of un-weighted samples to generate. Default: None (sets `nsamples` equal to the
         number of existing weighted samples).
        """

        if nsamples is None:
            nsamples = self.samples.shape[0]
        if method != "multinomial":
            raise ValueError("Exit code: Current available method: multinomial")
        multinomial_run = self.random_state.multinomial(nsamples, self.weights, size=1)[0]
        idx = []
        for j in range(self.samples.shape[0]):
            if multinomial_run[j] > 0:
                idx.extend([j for _ in range(multinomial_run[j])])
        self.unweighted_samples = self.samples[idx, :]

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        # log_pdf is provided
        if log_pdf_ is not None:
            if not callable(log_pdf_):
                raise TypeError("UQpy: log_pdf_target must be a callable")
            if args is None:
                args = ()
            evaluate_log_pdf = lambda x: log_pdf_(x, *args)
        elif pdf_ is not None:
            if not callable(pdf_):
                raise TypeError("UQpy: pdf_target must be a callable")
            if args is None:
                args = ()
            evaluate_log_pdf = lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],))))
        else:
            raise ValueError("UQpy: log_pdf_target or pdf_target should be provided.")
        return evaluate_log_pdf
