import logging

from beartype import beartype

from UQpy.sampling.input_data.ISInput import ISInput
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.utilities.Utilities import process_random_state
from UQpy.distributions import Distribution
import numpy as np


class ImportanceSampling:

    # Last Modified: 10/05/2020 by Audrey Olivier
    @beartype
    def __init__(self, is_input: ISInput, samples_number: PositiveInteger = None):
        """
        Sample from a user-defined target density using importance sampling.

        :param ISInput is_input: Object that contains input data to the :class:`ImportanceSampling` class.
         (See :class:`.ISInput`)
        :param samples_number: Number of samples to generate - see :meth:`run` method. If not `None`, the `run` method
         is called when the object is created. Default is None.
        """
        # Initialize proposal: it should have an rvs and log pdf or pdf method
        self.proposal = is_input.proposal
        if not isinstance(self.proposal, Distribution):
            raise TypeError("UQpy: The proposal should be of type Distribution.")
        if not hasattr(self.proposal, "rvs"):
            raise AttributeError("UQpy: The proposal should have an rvs method")
        if not hasattr(self.proposal, "log_pdf"):
            if not hasattr(self.proposal, "pdf"):
                raise AttributeError(
                    "UQpy: The proposal should have a log_pdf or pdf method"
                )
            self.proposal.log_pdf = lambda x: np.log(
                np.maximum(self.proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],)))
            )
        # self._pdf_target = is_input.pdf_target
        # self._log_pdf_target = is_input.log_pdf_target
        # self._args_target = is_input.args_target
        # Initialize target
        self.evaluate_log_target = self._preprocess_target(
            log_pdf_=is_input.log_pdf_target,
            pdf_=is_input.pdf_target,
            args=is_input.args_target,
        )

        self.logger = logging.getLogger(__name__)
        self.random_state = process_random_state(is_input.random_state)

        # Initialize the samples and weights
        self.samples = None
        self.unnormalized_log_weights = None
        self.weights = None
        self.unweighted_samples = None

        # Run IS if nsamples is provided
        if samples_number is not None and samples_number != 0:
            self.run(samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):
        """
        Generate and weight samples.

        This function samples from the proposal and appends samples to existing ones (if any). It then weights the
        samples as log_w_unnormalized) = log(target)-log(proposal).

        :param samples_number: Number of weighted samples to generate.

        This function has no returns, but it updates the output attributes `samples`, `unnormalized_log_weights` and
        `weights` of the :class:`.ImportanceSampling` object.
        """

        self.logger.info("UQpy: Running Importance Sampling...")
        # Sample from proposal
        new_samples = self.proposal.rvs(
            nsamples=samples_number, random_state=self.random_state
        )
        # Compute un-scaled weights of new samples
        new_log_weights = self.evaluate_log_target(
            x=new_samples
        ) - self.proposal.log_pdf(x=new_samples)

        # Save samples and weights (append to existing if necessary)
        if self.samples is None:
            self.samples = new_samples
            self.unnormalized_log_weights = new_log_weights
        else:
            self.samples = np.concatenate([self.samples, new_samples], axis=0)
            self.unnormalized_log_weights = np.concatenate(
                [self.unnormalized_log_weights, new_log_weights], axis=0
            )

        # Take the exponential and normalize the weights
        weights = np.exp(
            self.unnormalized_log_weights - max(self.unnormalized_log_weights)
        )
        # note: scaling with max avoids having NaN of Inf when taking the exp
        sum_w = np.sum(weights, axis=0)
        self.weights = weights / sum_w
        self.logger.info("UQpy: Importance Sampling performed successfully")

        # If a set of unweighted samples exist, delete them as they are not representative of the distribution anymore
        if self.unweighted_samples is not None:
            self.logger.info(
                "UQpy: unweighted samples are being deleted, call the resample method to regenerate them"
            )
            self.unweighted_samples = None

    def resample(self, method="multinomial", samples_number=None):
        """
        Resample to get a set of un-weighted samples that represent the target pdf.

        Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
        plotting for instance.

        The :meth:`resample` method is not called automatically when instantiating the :class:`.ImportanceSampling`
        class or when invoking its :meth:`run` method.

        :param method: Resampling method, as of V3 only multinomial resampling is supported. Default: 'multinomial'.
        :param samples_number: Number of un-weighted samples to generate. Default: None (sets `nsamples` equal to the
         number of existing weighted samples).

        The method has no returns, but it computes the following attribute of the :class:`ImportanceSampling` object.

        * **unweighted_samples** (`ndarray`)
            Un-weighted samples that represent the target pdf, `ndarray` of shape (nsamples, dimension)
        """

        if samples_number is None:
            samples_number = self.samples.shape[0]
        if method == "multinomial":
            multinomial_run = self.random_state.multinomial(
                samples_number, self.weights, size=1
            )[0]
            idx = list()
            for j in range(self.samples.shape[0]):
                if multinomial_run[j] > 0:
                    idx.extend([j for _ in range(multinomial_run[j])])
            self.unweighted_samples = self.samples[idx, :]
        else:
            raise ValueError("Exit code: Current available method: multinomial")

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x.

        :param log_pdf_: Log of the target density function from which to draw random samples. Either
         pdf_target or log_pdf_target must be provided
        :param pdf_: Target density function from which to draw random samples.
        :param args: Positional arguments of the pdf target
        :return: Callable that computes the log of the target density function
        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = lambda x: log_pdf_(x, *args)
            else:
                raise TypeError("UQpy: log_pdf_target must be a callable")
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = lambda x: np.log(
                    np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))
                )
            else:
                raise TypeError("UQpy: pdf_target must be a callable")
        else:
            raise ValueError("UQpy: log_pdf_target or pdf_target should be provided.")
        return evaluate_log_pdf
