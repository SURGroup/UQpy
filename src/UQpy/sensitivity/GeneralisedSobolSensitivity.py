"""

The GeneralisedSobol class computes the generalised Sobol indices for a given
multi-ouput model. The class is based on the work of [1]_ and [2]_.

Additionally, we can compute the confidence intervals for the Sobol indices 
using bootstrapping [3]_.

References
----------

 .. [1] Gamboa F, Janon A, Klein T, Lagnoux A, others.
        Sensitivity analysis for multidimensional and functional outputs.
        Electronic journal of statistics 2014; 8(1): 575-603.

 .. [2] Alexanderian A, Gremaud PA, Smith RC. Variance-based sensitivity 
        analysis for time-dependent processes. Reliability engineering 
        & system safety 2020; 196: 106722.

.. [3] Jeremy Orloff and Jonathan Bloom (2014), Bootstrap confidence intervals, 
       Introduction to Probability and Statistics, MIT OCW. 

"""

import logging

import numpy as np

from typing import Union
from beartype import beartype

from UQpy.sensitivity.baseclass.Sensitivity import Sensitivity
from UQpy.sensitivity.baseclass.PickFreeze import generate_pick_freeze_samples
from UQpy.utilities.UQpyLoggingFormatter import UQpyLoggingFormatter
from UQpy.utilities.ValidationTypes import (
    PositiveFloat,
    PositiveInteger,
    NumpyFloatArray,
    NumpyIntArray,
)


class GeneralisedSobolSensitivity(Sensitivity):
    """
    Compute the generalised Sobol indices for models with multiple outputs
    (vector-valued response) using the Pick-and-Freeze method.

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
        self, runmodel_object, dist_object, random_state=None, **kwargs
    ) -> None:

        super().__init__(runmodel_object, dist_object, random_state, **kwargs)

        # Create logger with the same name as the class
        self.logger = logging.getLogger(__name__)

        self.generalized_first_order_indices = None
        "Generalised first order Sobol indices, :any:`numpy.ndarray` of shape (n_variables, 1)"

        self.generalized_total_order_indices = None
        "Generalised total order Sobol indices, :any:`numpy.ndarray` of shape (n_variables, 1)"

        self.n_samples = None
        "Number of samples used to compute the sensitivity indices, :class:`int`"

        self.n_variables = None
        "Number of model input variables, :class:`int`"

    @beartype
    def run(
        self,
        n_samples: PositiveInteger = 1_000,
        n_bootstrap_samples: PositiveInteger = None,
        confidence_level: PositiveFloat = 0.95,
    ):

        """
        Compute the generalised Sobol indices for models with multiple outputs
        (vector-valued response) using the Pick-and-Freeze method.

        :param n_samples: Number of samples used to compute the sensitivity indices. \
            Default is 1,000.

        :param n_bootstrap_samples: Number of bootstrap samples used to compute the \
            confidence intervals. Default is :any:`None`.

        :param confidence_level: Confidence level used to compute the confidence \
            intervals. Default is 0.95.
        """

        # Check n_samples data type
        self.n_samples = n_samples
        if not isinstance(self.n_samples, int):
            raise TypeError("UQpy: n_samples should be an integer")

        # Check num_bootstrap_samples data type
        if n_bootstrap_samples is None:
            self.logger.info("UQpy: num_bootstrap_samples is set to None, confidence intervals will not be computed.\n")

        elif not isinstance(n_bootstrap_samples, int):
            raise TypeError("UQpy: num_bootstrap_samples should be an integer.\n")
        ################## GENERATE SAMPLES ##################

        (A_samples, B_samples, C_i_generator, _,) = generate_pick_freeze_samples(
            self.dist_object, self.n_samples, self.random_state)

        self.logger.info("UQpy: Generated samples using the pick-freeze scheme.\n")

        self.n_variables = A_samples.shape[1]  # Number of variables

        ################# MODEL EVALUATIONS ####################

        A_model_evals = self._run_model(A_samples)  # shape: (n_samples, n_outputs)

        # if model output is vectorised,
        # shape retured by model is (n_samples, n_outputs, 1)
        # we need to reshape it to (n_samples, n_outputs)
        if A_model_evals.ndim == 3:
            A_model_evals = A_model_evals[:, :, 0]  # shape: (n_samples, n_outputs)

        self.logger.info("UQpy: Model evaluations A completed.\n")

        B_model_evals = self._run_model(B_samples)  # shape: (n_samples, n_outputs)

        # if model output is vectorised,
        # shape retured by model is (n_samples, n_outputs, 1)
        # we need to reshape it to (n_samples, n_outputs)
        if B_model_evals.ndim == 3:
            B_model_evals = B_model_evals[:, :, 0]  # shape: (n_samples, n_outputs)

        self.logger.info("UQpy: Model evaluations B completed.\n")

        self.n_outputs = A_model_evals.shape[1]

        # shape: (n_outputs, n_samples, n_variables)
        C_i_model_evals = np.zeros((self.n_outputs, self.n_samples, self.n_variables))

        for i, C_i in enumerate(C_i_generator):

            # if model output is vectorised,
            # shape retured by model is (n_samples, n_outputs, 1)
            # we need to reshape it to (n_samples, n_outputs)
            model_evals = self._run_model(C_i)

            if model_evals.ndim == 3:
                C_i_model_evals[:, :, i] = self._run_model(C_i)[:, :, 0].T
            else:
                C_i_model_evals[:, :, i] = model_evals.T

        self.logger.info("UQpy: Model evaluations C completed.\n")

        self.logger.info("UQpy: All model evaluations computed successfully.\n")

        ################## COMPUTE GENERALISED SOBOL INDICES ##################

        self.generalized_first_order_indices = self.compute_first_order_generalised_sobol_indices(
            A_model_evals, B_model_evals, C_i_model_evals)

        self.logger.info("UQpy: First order Generalised Sobol indices computed successfully.\n")

        self.generalized_total_order_indices = self.compute_total_order_generalised_sobol_indices(
            A_model_evals, B_model_evals, C_i_model_evals)

        self.logger.info("UQpy: Total order Generalised Sobol indices computed successfully.\n")


        ################## CONFIDENCE INTERVALS ####################

        if n_bootstrap_samples is not None:

            self.logger.info("UQpy: Computing confidence intervals ...\n")

            estimator_inputs = [
                A_model_evals,
                B_model_evals,
                C_i_model_evals,
            ]

            # First order generalised Sobol indices
            self.first_order_confidence_interval = self.bootstrapping(
                self.compute_first_order_generalised_sobol_indices,
                estimator_inputs,
                self.generalized_first_order_indices,
                n_bootstrap_samples,
                confidence_level,
            )

            self.logger.info(
                "UQpy: Confidence intervals for First order Generalised Sobol indices computed successfully.\n")

            # Total order generalised Sobol indices
            self.total_order_confidence_interval = self.bootstrapping(
                self.compute_total_order_generalised_sobol_indices,
                estimator_inputs,
                self.generalized_total_order_indices,
                n_bootstrap_samples,
                confidence_level,
            )

            self.logger.info(
                "UQpy: Confidence intervals for Total order Sobol Generalised indices computed successfully.\n")


    @staticmethod
    @beartype
    def compute_first_order_generalised_sobol_indices(
        A_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        B_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        C_i_model_evals: Union[NumpyFloatArray, NumpyIntArray],
    ):

        """
        Compute the generalised Sobol indices for models with multiple outputs.

        :param A_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, n_outputs)`.
        :param B_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, n_outputs)`.
        :param C_i_model_evals: Model evaluations, :class:`numpy.ndarray` of shape
        :code:`(n_outputs, n_samples, n_variables)`.

        :return: First order generalised Sobol indices, :class:`numpy.ndarray` of shape
        :code:`(n_outputs, n_variables)`.

        """

        num_vars = C_i_model_evals.shape[2]
        n_outputs = A_model_evals.shape[1]

        # store generalised Sobol indices
        gen_sobol_i = np.zeros((num_vars, 1))

        for i in range(num_vars):

            all_Y_i = A_model_evals.T  # shape: (n_outputs, n_samples)
            all_Y_i_tilde = B_model_evals.T  # shape: (n_outputs, n_samples)
            all_Y_i_u = C_i_model_evals[:, :, i]  # shape: (n_outputs, n_samples)

            # compute the mean using all model evaluations
            # shape: (n_outputs, 1)
            mean = (
                np.mean(all_Y_i, axis=1, keepdims=1)
                + np.mean(all_Y_i_u, axis=1, keepdims=1)
                + np.mean(all_Y_i_tilde, axis=1, keepdims=1)
            ) / 3

            # center the evaluations since mean is available
            all_Y_i = all_Y_i - mean
            all_Y_i_tilde = all_Y_i_tilde - mean
            all_Y_i_u = all_Y_i_u - mean

            # compute the variance matrix using all available model evaluations
            # shape: (n_outputs, n_outputs)
            C = (np.cov(all_Y_i) + np.cov(all_Y_i_u) + np.cov(all_Y_i_tilde)) / 3

            # compute covariance btw. RVs 'X' and 'Y'
            # shape: (2*n_outputs, 2*n_outputs)
            # It contains the following 4 block matrices:
            #   (1, 1) variance of 'X'
            #  *(1, 2) covariance between 'X' and 'Y' (a.k.a. cross-covariance)
            #   (2, 1) covariance between 'Y' and 'X' (a.k.a. cross-covariance)
            #   (2, 2) variance of 'Y'
            _cov_1 = np.cov(all_Y_i_u, all_Y_i)  # for first order indices

            # We need the cross-covariance between 'X' and 'Y'
            # Extract *(1, 2) (upper right block)
            # shape: (n_outputs, n_outputs)
            C_u = _cov_1[0:n_outputs, n_outputs : 2 * n_outputs]

            denominator = np.trace(C)

            # Generalised Sobol indices
            gen_sobol_i[i] = np.trace(C_u) / denominator

        return gen_sobol_i

    @staticmethod
    @beartype
    def compute_total_order_generalised_sobol_indices(
        A_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        B_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        C_i_model_evals: Union[NumpyFloatArray, NumpyIntArray],
    ):

        """
        Compute the generalised Sobol indices for models with multiple outputs.

        :param A_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, n_outputs)`.
        :param B_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, n_outputs)`.
        :param C_i_model_evals: Model evaluations, :class:`numpy.ndarray` of shape
        :code:`(n_outputs, n_samples, n_variables)`.

        :return: Total order generalised Sobol indices, :class:`numpy.ndarray` of shape
        :code:`(n_outputs, n_variables)`.

        """

        num_vars = C_i_model_evals.shape[2]
        n_outputs = A_model_evals.shape[1]

        # store generalised Sobol indices
        gen_sobol_total_i = np.zeros((num_vars, 1))

        for i in range(num_vars):

            all_Y_i = A_model_evals.T  # shape: (n_outputs, n_samples)
            all_Y_i_tilde = B_model_evals.T  # shape: (n_outputs, n_samples)
            all_Y_i_u = C_i_model_evals[:, :, i]  # shape: (n_outputs, n_samples)

            # compute the mean using all model evaluations
            # shape: (n_outputs, 1)
            mean = (
                np.mean(all_Y_i, axis=1, keepdims=1)
                + np.mean(all_Y_i_u, axis=1, keepdims=1)
                + np.mean(all_Y_i_tilde, axis=1, keepdims=1)
            ) / 3

            # center the evaluations since mean is available
            all_Y_i = all_Y_i - mean
            all_Y_i_tilde = all_Y_i_tilde - mean
            all_Y_i_u = all_Y_i_u - mean

            # compute the variance matrix using all available model evaluations
            # shape: (n_outputs, n_outputs)
            C = (np.cov(all_Y_i) + np.cov(all_Y_i_u) + np.cov(all_Y_i_tilde)) / 3

            # compute covariance btw. RVs 'X' and 'Y'
            # shape: (2*n_outputs, 2*n_outputs)
            # It contains the following 4 block matrices:
            #   (1, 1) variance of 'X'
            #  *(1, 2) covariance between 'X' and 'Y' (a.k.a. cross-covariance)
            #   (2, 1) covariance between 'Y' and 'X' (a.k.a. cross-covariance)
            #   (2, 2) variance of 'Y'
            _cov_2 = np.cov(all_Y_i_u, all_Y_i_tilde)  # for total order indices

            # We need the cross-covariance between 'X' and 'Y'
            # Extract *(1, 2) (upper right block)
            # shape: (n_outputs, n_outputs)
            C_u_tilde = _cov_2[0:n_outputs, n_outputs : 2 * n_outputs]
            denominator = np.trace(C)

            # Generalised Sobol indices
            gen_sobol_total_i[i] = 1 - np.trace(C_u_tilde) / denominator

        return gen_sobol_total_i
