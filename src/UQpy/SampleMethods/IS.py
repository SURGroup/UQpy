from UQpy.Distributions import Distribution
import numpy as np
########################################################################################################################
########################################################################################################################
#                                        Generating random samples inside a Simplex
########################################################################################################################

class IS:
    """
    Sample from a user-defined target density using importance sampling.


    **Inputs:**

    * **nsamples** (`int`):
        Number of samples to generate - see ``run`` method. If not `None`, the `run` method is called when the object is
        created. Default is None.

    * **pdf_target** (callable):
        Callable that evaluates the pdf of the target distribution. Either log_pdf_target or pdf_target must be
        specified (the former is preferred).

    * **log_pdf_target** (callable)
        Callable that evaluates the log-pdf of the target distribution. Either log_pdf_target or pdf_target must be
        specified (the former is preferred).

    * **args_target** (`tuple`):
        Positional arguments of the target log_pdf / pdf callable.

    * **proposal** (``Distribution`` object):
        Proposal to sample from. This ``UQpy.Distributions`` object must have an rvs method and a log_pdf (or pdf)
        method.

    * **verbose** (`boolean`)
        Set ``verbose = True`` to print status messages to the terminal during execution.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.


    **Attributes:**

    * **samples** (`ndarray`):
        Set of samples, `ndarray` of shape (nsamples, dim)

    * **unnormalized_log_weights** (`ndarray`)
        Unnormalized log weights, i.e., log_w(x) = log_target(x) - log_proposal(x), `ndarray` of shape (nsamples, )

    * **weights** (`ndarray`):
        Importance weights, weighted so that they sum up to 1, `ndarray` of shape (nsamples, )

    * **unweighted_samples** (`ndarray`):
        Set of un-weighted samples (useful for instance for plotting), computed by calling the `resample` method

    **Methods:**
    """
    # Last Modified: 10/05/2020 by Audrey Olivier
    def __init__(self, nsamples=None, pdf_target=None, log_pdf_target=None, args_target=None,
                 proposal=None, verbose=False, random_state=None):
        # Initialize proposal: it should have an rvs and log pdf or pdf method
        self.proposal = proposal
        if not isinstance(self.proposal, Distribution):
            raise TypeError('UQpy: The proposal should be of type Distribution.')
        if not hasattr(self.proposal, 'rvs'):
            raise AttributeError('UQpy: The proposal should have an rvs method')
        if not hasattr(self.proposal, 'log_pdf'):
            if not hasattr(self.proposal, 'pdf'):
                raise AttributeError('UQpy: The proposal should have a log_pdf or pdf method')
            self.proposal.log_pdf = lambda x: np.log(np.maximum(self.proposal.pdf(x),
                                                                10 ** (-320) * np.ones((x.shape[0],))))

        # Initialize target
        self.evaluate_log_target = self._preprocess_target(log_pdf_=log_pdf_target, pdf_=pdf_target, args=args_target)

        self.verbose = verbose
        self.random_state = random_state
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        # Initialize the samples and weights
        self.samples = None
        self.unnormalized_log_weights = None
        self.weights = None
        self.unweighted_samples = None

        # Run IS if nsamples is provided
        if nsamples is not None and nsamples != 0:
            self.run(nsamples)

    def run(self, nsamples):
        """
        Generate and weight samples.

        This function samples from the proposal and appends samples to existing ones (if any). It then weights the
        samples as log_w_unnormalized) = log(target)-log(proposal).

        **Inputs:**

        * **nsamples** (`int`)
            Number of weighted samples to generate.

        * **Output/Returns:**

        This function has no returns, but it updates the output attributes `samples`, `unnormalized_log_weights` and
        `weights` of the ``IS`` object.
        """

        if self.verbose:
            print('UQpy: Running Importance Sampling...')
        # Sample from proposal
        new_samples = self.proposal.rvs(nsamples=nsamples, random_state=self.random_state)
        # Compute un-scaled weights of new samples
        new_log_weights = self.evaluate_log_target(x=new_samples) - self.proposal.log_pdf(x=new_samples)

        # Save samples and weights (append to existing if necessary)
        if self.samples is None:
            self.samples = new_samples
            self.unnormalized_log_weights = new_log_weights
        else:
            self.samples = np.concatenate([self.samples, new_samples], axis=0)
            self.unnormalized_log_weights = np.concatenate([self.unnormalized_log_weights, new_log_weights], axis=0)

        # Take the exponential and normalize the weights
        weights = np.exp(self.unnormalized_log_weights - max(self.unnormalized_log_weights))
        # note: scaling with max avoids having NaN of Inf when taking the exp
        sum_w = np.sum(weights, axis=0)
        self.weights = weights / sum_w
        if self.verbose:
            print('UQpy: Importance Sampling performed successfully')

        # If a set of unweighted samples exist, delete them as they are not representative of the distribution anymore
        if self.unweighted_samples is not None:
            if self.verbose:
                print('UQpy: unweighted samples are being deleted, call the resample method to regenerate them')
            self.unweighted_samples = None

    # def resample(self, method='multinomial', nsamples=None):
    #     """
    #     Resample to get a set of un-weighted samples that represent the target pdf.
    #
    #     Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
    #     plotting for instance.
    #
    #     **Inputs:**
    #
    #     * **method** (`str`)
    #         Resampling method, as of V3 only multinomial resampling is supported. Default: 'multinomial'.
    #     * **nsamples** (`int`)
    #         Number of un-weighted samples to generate. Default: None (same number of samples is generated as number of
    #         existing samples).
    #
    #     **Output/Returns:**
    #
    #     * (`ndarray`)
    #         Un-weighted samples that represent the target pdf, `ndarray` of shape (nsamples, dimension)
    #
    #     """
    #     from .Utilities import resample
    #     return resample(self.samples, self.weights, method=method, size=nsamples)

    def resample(self, method='multinomial', nsamples=None):
        """
        Resample to get a set of un-weighted samples that represent the target pdf.

        Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
        plotting for instance.

        The ``resample`` method is not called automatically when instantiating the ``IS`` class or when invoking its
        ``run`` method.

        **Inputs:**

        * **method** (`str`)
            Resampling method, as of V3 only multinomial resampling is supported. Default: 'multinomial'.
        * **nsamples** (`int`)
            Number of un-weighted samples to generate. Default: None (sets `nsamples` equal to the number of
            existing weighted samples).

        **Output/Returns:**

        The method has no returns, but it computes the following attribute of the ``IS`` object.

        * **unweighted_samples** (`ndarray`)
            Un-weighted samples that represent the target pdf, `ndarray` of shape (nsamples, dimension)

        """

        if nsamples is None:
            nsamples = self.samples.shape[0]
        if method == 'multinomial':
            multinomial_run = self.random_state.multinomial(nsamples, self.weights, size=1)[0]
            idx = list()
            for j in range(self.samples.shape[0]):
                if multinomial_run[j] > 0:
                    idx.extend([j for _ in range(multinomial_run[j])])
            self.unweighted_samples = self.samples[idx, :]
        else:
            raise ValueError('Exit code: Current available method: multinomial')

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x.

        **Inputs:**

        * log_pdf_ ((list of) callables): Log of the target density function from which to draw random samples. Either
          pdf_target or log_pdf_target must be provided
        * pdf_ ((list of) callables): Target density function from which to draw random samples.
        * args (tuple): Positional arguments of the pdf target

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function

        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf_(x, *args))
            else:
                raise TypeError('UQpy: log_pdf_target must be a callable')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('UQpy: pdf_target must be a callable')
        else:
            raise ValueError('UQpy: log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf