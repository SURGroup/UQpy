"""
Computing the Cramér-von Mises sensitivity indices.

References
----------

.. [1] Gamboa, F., Klein, T., & Lagnoux, A. (2018). Sensitivity Analysis
       Based on Cramér-von Mises Distance. SIAM/ASA Journal on Uncertainty
       Quantification, 6(2), 522-548. doi:10.1137/15M1025621

.. [2] Gamboa, F., Gremaud, P., Klein, T., & Lagnoux, A. (2020). Global
       Sensitivity Analysis: a new generation of mighty estimators based on
       rank statistics. arXiv [math.ST]. http://arxiv.org/abs/2003.01772

"""

import logging
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.sensitivity.baseclass.Sensitivity import Sensitivity
from UQpy.sensitivity.baseclass.PickFreeze import generate_pick_freeze_samples
from UQpy.sensitivity.SobolSensitivity import compute_first_order as compute_first_order_sobol
from UQpy.sensitivity.SobolSensitivity import compute_total_order as compute_total_order_sobol
from UQpy.utilities.UQpyLoggingFormatter import UQpyLoggingFormatter
from UQpy.utilities.ValidationTypes import (
    PositiveInteger,
    PositiveFloat,
    NumpyFloatArray,
    NumpyIntArray,
)


# TODO: Sampling strategies


class CramerVonMisesSensitivity(Sensitivity):
    """
    Compute the Cramér-von Mises indices.

    Currently only available for models with scalar output.

    :param runmodel_object: The computational model. It should be of type :class:`.RunModel`. \
        The output QoI can be a scalar or vector of length :code:`ny`, then the sensitivity \
        indices of all :code:`ny` outputs are computed independently.

    :param distributions: List of :class:`.Distribution` objects corresponding to each \
        random variable, or :class:`.JointIndependent` object \
        (multivariate RV with independent marginals).

    :param random_state: Random seed used to initialize the pseudo-random number \
        generator. Default is :any:`None`.

    **Methods:**
    """

    def __init__(
        self, runmodel_object, dist_object, random_state=None
    ) -> None:

        super().__init__(runmodel_object, dist_object, random_state=random_state)

        # Create logger with the same name as the class
        self.logger = logging.getLogger(__name__)

        self.first_order_CramerVonMises_indices = None
        "First order Cramér-von Mises indices, :class:`numpy.ndarray` of shape :code:`(n_variables, 1)`"

        self.confidence_interval_CramerVonMises = None
        "Confidence intervals of the first order Cramér-von Mises indices, :class:`numpy.ndarray` " \
        "of shape :code:`(n_variables, 2)`"

        self.first_order_sobol_indices = None
        "First order Sobol indices computed using the pick-and-freeze samples, :class:`numpy.ndarray` " \
        "of shape :code:`(n_variables, 1)`"

        self.total_order_sobol_indices = None
        "Total order Sobol indices computed using the pick-and-freeze samples, :class:`numpy.ndarray` " \
        "of shape :code:`(n_variables, 1)`"

        self.n_samples = None
        "Number of samples used to compute the Cramér-von Mises indices, :class:`int`"

        self.n_variables = None
        "Number of input random variables, :class:`int`"

    @beartype
    def run(
        self,
        n_samples: PositiveInteger = 1_000,
        estimate_sobol_indices: bool = False,
        num_bootstrap_samples: PositiveInteger = None,
        confidence_level: PositiveFloat = 0.95,
        disable_CVM_indices: bool = False,
    ):

        """
        Compute the Cramér-von Mises indices.

        :param n_samples: Number of samples used to compute the Cramér-von Mises indices. \
            Default is 1,000.

        :param estimate_sobol_indices: If :code:`True`, the Sobol indices are estimated \
            using the pick-and-freeze samples.

        :param num_bootstrap_samples: Number of bootstrap samples used to estimate the \
            Sobol indices. Default is :any:`None`.

        :param confidence_level: Confidence level used to compute the confidence \
            intervals of the Cramér-von Mises indices.

        :param disable_CVM_indices: If :code:`True`, the Cramér-von Mises indices \
            are not computed.
        """

        # Check nsamples
        self.n_samples = n_samples
        if not isinstance(self.n_samples, int):
            raise TypeError("UQpy: nsamples should be an integer")

        # Check num_bootstrap_samples data type
        if num_bootstrap_samples is None:
            self.logger.info("UQpy: num_bootstrap_samples is set to None, confidence intervals will not be computed.\n")

        elif not isinstance(num_bootstrap_samples, int):
            raise TypeError("UQpy: num_bootstrap_samples should be an integer.\n")
        ################## GENERATE SAMPLES ##################

        A_samples, W_samples, C_i_generator, _ = generate_pick_freeze_samples(
            self.dist_object, self.n_samples, self.random_state)

        self.logger.info("UQpy: Generated samples using the pick-freeze scheme.\n")

        ################# MODEL EVALUATIONS ####################

        A_model_evals = self._run_model(A_samples).reshape(-1, 1)

        self.logger.info("UQpy: Model evaluations A completed.\n")

        W_model_evals = self._run_model(W_samples).reshape(-1, 1)

        self.logger.info("UQpy: Model evaluations W completed.\n")

        self.n_variables = A_samples.shape[1]

        C_i_model_evals = np.zeros((self.n_samples, self.n_variables))

        for i, C_i in enumerate(C_i_generator):
            C_i_model_evals[:, i] = self._run_model(C_i).ravel()

        self.logger.info("UQpy: Model evaluations C completed.\n")

        self.logger.info("UQpy: All model evaluations computed successfully.\n")

        ################## COMPUTE CVM INDICES ##################

        # flag is used to disable computation of
        # CVM indices during testing
        if not disable_CVM_indices:
            # Compute the Cramér-von Mises indices
            self.first_order_CramerVonMises_indices = self.pick_and_freeze_estimator(
                A_model_evals, W_model_evals, C_i_model_evals)

            self.logger.info("UQpy: Cramér-von Mises indices computed successfully.\n")


        ################# COMPUTE CONFIDENCE INTERVALS ##################

        if num_bootstrap_samples is not None:

            self.logger.info("UQpy: Computing confidence intervals ...\n")

            estimator_inputs = [
                A_model_evals,
                W_model_evals,
                C_i_model_evals,
            ]

            self.confidence_interval_CramerVonMises = self.bootstrapping(
                self.pick_and_freeze_estimator,
                estimator_inputs,
                self.first_order_CramerVonMises_indices,
                num_bootstrap_samples,
                confidence_level,
            )

            self.logger.info("UQpy: Confidence intervals for Cramér-von Mises indices computed successfully.\n")


        ################## COMPUTE SOBOL INDICES ##################

        if estimate_sobol_indices:

            self.logger.info("UQpy: Computing First order Sobol indices ...\n")

            # extract shape
            _shape = C_i_model_evals.shape

            # convert C_i_model_evals to 3D array
            # with n_outputs=1 in first dimension
            n_outputs = 1
            C_i_model_evals = C_i_model_evals.reshape((n_outputs, *_shape))

            self.first_order_sobol_indices = compute_first_order_sobol(
                A_model_evals, W_model_evals, C_i_model_evals)

            self.logger.info("UQpy: First order Sobol indices computed successfully.\n")

            self.total_order_sobol_indices = compute_total_order_sobol(
                A_model_evals, W_model_evals, C_i_model_evals)

            self.logger.info("UQpy: Total order Sobol indices computed successfully.\n")


    @staticmethod
    @beartype
    def indicator_function(Y: Union[NumpyFloatArray, NumpyIntArray], w: float):
        """
        Vectorized version of the indicator function.

        .. math::
           \mathbb{I}(Y,W) = \mathbf{1}_{Y \leq W}

        **Inputs:**

        * **Y** (`ndarray`):
            Array of values of the random variable.
            Shape: `(N, 1)`

        * **w** (`float`):
            Value to compare with the array.

        **Outputs:**

        * **indicator** (`ndarray`):
            Array of integers with truth values.
            Shape: `(N, 1)`

        """
        return (Y <= w).astype(int)

    @beartype
    def pick_and_freeze_estimator(
        self,
        A_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        W_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        C_i_model_evals: Union[NumpyFloatArray, NumpyIntArray],
    ):

        """
        Compute the first order Cramér-von Mises indices
        using the Pick-and-Freeze estimator.

        **Inputs**

        * **A_model_evals** (`np.array`):
            Shape: `(n_samples, 1)`

        * **W_model_evals** (`np.array`):
            Shape: `(n_samples, 1)`

        * **C_i_model_evals** (`np.array`):
            Shape: `(n_samples, n_variables)`

        **Outputs**

        * **First_order_CVM** (`np.array`):
            Shape: `(n_variables)`

        """

        ## **Notes**

        # Implementation using 2 `for` loops. This is however
        # faster than the vectorized version which has only 1 `for` loop.

        # For N = 50_000 runs
        # With 2 `for` loops: 26.75 seconds (this implementation)
        # With 1 `for` loops: 62.42 seconds (vectorized implementation)

        # Possible improvements:
        # Check indicator function run time using a profiler
        # as it results in an `N` x `N` array.
        # Q. Does it use a for loop under the hood?
        # Computations such as `np.sum` and `np.mean`
        # are handled by numpy so they are fast.
        # (This should however be faster for small `N`, e.g. N=10_000)

        N = self.n_samples
        m = self.n_variables

        # Model evaluations
        f_A = A_model_evals.ravel()
        f_W = W_model_evals.ravel()
        f_C_i = C_i_model_evals

        # Store CramérvonMises indices
        first_order_indices = np.zeros((m, 1))

        # Compute Cramér-von Mises indices
        for i in range(m):
            sum_numerator = 0
            sum_denominator = 0

            for k in range(N):

                term_1 = self.indicator_function(f_A, f_W[k])
                term_2 = self.indicator_function(f_C_i[:, i], f_W[k])

                mean_sum = (1 / (2 * N)) * np.sum(term_1 + term_2)
                mean_product = (1 / N) * np.sum(term_1 * term_2)

                sum_numerator += mean_product - mean_sum**2
                sum_denominator += mean_sum - mean_sum**2

            first_order_indices[i] = sum_numerator / sum_denominator

        return first_order_indices
