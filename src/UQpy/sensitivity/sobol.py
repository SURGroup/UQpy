"""

The Sobol class computes the Sobol indices for single output and multi-output
models. The Sobol indices can be computed using various pick-and-freeze 
schemes.

The schemes implemented are listed below:

# First order indices: 
- Sobol1993 [1]: Requires n_samples*(num_vars + 1) model evaluations
- Saltelli2002 [3]: Requires n_samples*(2*num_vars + 1) model evaluations
- Janon2014 [4]: Requires n_samples*(num_vars + 1) model evaluations

# Second order indices:
- Saltelli2002 [3]: Requires n_samples*(2*num_vars + 1) model evaluations

# Total order indices:
- Homma1996: Requires n_samples*(num_vars + 1) model evaluations
- Saltelli2002 [3]: Requires n_samples*(2*num_vars + 1) model evaluations

For more details on "Saltelli2002" refer to [3].
    
Note: Apart from second order indices, the Saltelli2002 scheme provides 
      more accurate estimates of all indices, as opposed to Homma1996 or Sobol1993. 
      Because this method efficiently utilizes the higher number of model evaluations.

Additionally, we can compute the confidence intervals for the Sobol indices 
using bootstrapping [2].


References
----------

.. [1] Sobol, I.M. (1993) Sensitivity Estimates for Nonlinear Mathematical Models.  
       Mathematical Modelling and Computational Experiments, 4, 407-414.

.. [2] Jeremy Orloff and Jonathan Bloom (2014), Bootstrap confidence intervals, 
       Introduction to Probability and Statistics, MIT OCW. 

.. [3] Saltelli, A. (2002). Making best use of model evaluations to
       compute sensitivity indices.

.. [4] Janon, Alexander; Klein, Thierry; Lagnoux, Agnes; Nodet, Maëlle; 
       Prior, Clementine. Asymptotic normality and efficiency of two Sobol index 
       estimators. ESAIM: Probability and Statistics, Volume 18 (2014), pp. 342-364. 
       doi:10.1051/ps/2013040. http://www.numdam.org/articles/10.1051/ps/2013040/

"""

import math
import logging
import itertools

import numpy as np

from UQpy.sensitivity.baseclass.sensitivity import Sensitivity
from UQpy.sensitivity.baseclass.pickfreeze import generate_pick_freeze_samples
from UQpy.utilities.UQpyLoggingFormatter import UQpyLoggingFormatter

# TODO: Sampling strategies


class Sobol(Sensitivity):
    """
    Compute Sobol sensitivity indices using the pick
    and freeze algorithm. For models with multiple outputs
    (vector-valued response), the sensitivity indices are computed for each
    output separately.
    For time-series models, the sensitivity indices are computed for each
    time instant separately. (Pointwise-in-time Sobol indices)

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
        self.logger.setLevel(logging.ERROR)
        frmt = UQpyLoggingFormatter()

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setFormatter(frmt)

        # add the handler to the logger
        self.logger.addHandler(ch)

        self.sobol_i = None
        "First order Sobol indices, :class:`numpy.ndarray` of shape `(num_vars, n_outputs)`"

        self.sobol_total_i = None
        "Total order Sobol indices, :class:`numpy.ndarray` of shape `(num_vars, n_outputs)`"

        self.sobol_ij = None
        "Second order Sobol indices, :class:`numpy.ndarray` of shape `(num_second_order_terms, n_outputs)`"

        self.CI_sobol_i = None
        "Confidence intervals for the first order Sobol indices, :class:`numpy.ndarray` of shape `(num_vars, 2)`"

        self.CI_sobol_total_i = None
        "Confidence intervals for the total order Sobol indices, :class:`numpy.ndarray` of shape `(num_vars, 2)`"

        self.CI_sobol_ij = None
        "Confidence intervals for the second order Sobol indices, :class:`numpy.ndarray` of shape `(num_second_order_terms, 2)`"

        self.n_samples = None
        "Number of samples used to compute the sensitivity indices, :class:`int`"

        self.num_vars = None
        "Number of model input variables, :class:`int`"

        self.multioutput = None
        "True if the model has multiple outputs, :class:`bool`"

    def run(
        self,
        n_samples=1_000,
        num_bootstrap_samples=None,
        confidence_level=0.95,
        estimate_second_order=False,
        first_order_scheme="Janon2014",
        total_order_scheme="Homma1996",
        second_order_scheme="Saltelli2002",
    ):

        """
        Compute the sensitivity indices and confidence intervals.

        :param n_samples: Number of samples used to compute the sensitivity indices. \
            Default is 1,000.

        :param num_bootstrap_samples: Number of bootstrap samples used to compute the \
            confidence intervals. Default is :any:`None`.

        :param confidence_interval: Confidence level used to compute the confidence \
            intervals. Default is 0.95.

        :param estimate_second_order: If True, the second order Sobol indices are \
            estimated. Default is False.

        :param first_order_scheme: Scheme used to compute the first order Sobol \
            indices. Default is "Janon2014".

        :param total_order_scheme: Scheme used to compute the total order Sobol \
            indices. Default is "Homma1996".

        :param second_order_scheme: Scheme used to compute the second order \
            Sobol indices. Default is "Saltelli2002".

        :return: A :class:`dict` with the following keys: \
            :code:`sobol_i` of shape :code:`(num_vars, 1)`, \
            :code:`sobol_total_i` of shape :code:`(num_vars, 1)`, \
            :code:`sobol_ij` of shape :code:`(num_second_order_terms, 1)`, \
            :code:`CI_sobol_i` of shape :code:`(num_vars, 2)`, \
            if multioutput: Shape: `(n_outputs, num_vars, 2)`, \
            :code:`CI_sobol_total_i` of shape :code:`(num_vars, 2)`, \
            if multioutput: Shape: `(n_outputs, num_vars, 2)`, \
            :code:`CI_sobol_ij` of shape :code:`(num_second_order_terms, 2)`
            if multioutput: Shape: `(n_outputs, num_second_order_terms, 2)`, \

        """
        # Check n_samples data type
        self.n_samples = n_samples
        if not isinstance(self.n_samples, int):
            raise TypeError("UQpy: n_samples should be an integer.")

        # Check num_bootstrap_samples data type
        if num_bootstrap_samples is not None:
            if not isinstance(num_bootstrap_samples, int):
                raise TypeError("UQpy: num_bootstrap_samples should be an integer.")
        elif num_bootstrap_samples is None:
            self.logger.info(
                "UQpy: num_bootstrap_samples is set to None, confidence intervals will not be computed."
            )

        ################## GENERATE SAMPLES ##################

        (
            A_samples,
            B_samples,
            C_i_generator,
            D_i_generator,
        ) = generate_pick_freeze_samples(
            self.dist_object, self.n_samples, self.random_state
        )

        self.logger.info("UQpy: Generated samples using the pick-freeze scheme.")

        self.num_vars = A_samples.shape[1]  # Number of variables

        ################# MODEL EVALUATIONS ####################

        A_model_evals = self._run_model(A_samples)  # shape: (n_samples, n_outputs)

        self.logger.info("UQpy: Model evaluations A completed.")

        B_model_evals = self._run_model(B_samples)  # shape: (n_samples, n_outputs)

        self.logger.info("UQpy: Model evaluations B completed.")

        # Check the number of outputs of the model
        try:
            self.n_outputs = A_model_evals.shape[1]
        except:
            self.n_outputs = 1

        # multioutput flag
        self.multioutput = True if self.n_outputs > 1 else False

        if not self.multioutput:
            A_model_evals = A_model_evals.reshape(-1, 1)
            B_model_evals = B_model_evals.reshape(-1, 1)

        C_i_model_evals = np.zeros((self.n_outputs, self.n_samples, self.num_vars))

        for i, C_i in enumerate(C_i_generator):
            C_i_model_evals[:, :, i] = self._run_model(C_i).T

        self.logger.info("UQpy: Model evaluations C completed.")

        # Compute D_i_model_evals only if needed
        if estimate_second_order or total_order_scheme == "Saltelli2002":

            D_i_model_evals = np.zeros((self.n_outputs, self.n_samples, self.num_vars))

            for i, D_i in enumerate(D_i_generator):
                D_i_model_evals[:, :, i] = self._run_model(D_i).T

            self.logger.info("UQpy: Model evaluations D completed.")

        else:
            D_i_model_evals = None

        self.logger.info("UQpy: All model evaluations computed successfully.")

        ######################### STORAGE ########################

        # Create dictionary to store the sensitivity indices
        computed_indices = {}

        ################## COMPUTE SOBOL INDICES ##################

        # First order Sobol indices
        self.sobol_i = compute_first_order(
            A_model_evals,
            B_model_evals,
            C_i_model_evals,
            D_i_model_evals,
            scheme=first_order_scheme,
        )

        self.logger.info("UQpy: First order Sobol indices computed successfully.")

        computed_indices["sobol_i"] = self.sobol_i

        # Total order Sobol indices
        self.sobol_total_i = compute_total_order(
            A_model_evals,
            B_model_evals,
            C_i_model_evals,
            D_i_model_evals,
            scheme=total_order_scheme,
        )

        self.logger.info("UQpy: Total order Sobol indices computed successfully.")

        computed_indices["sobol_total_i"] = self.sobol_total_i

        if estimate_second_order:

            # Second order Sobol indices
            self.sobol_ij = compute_second_order(
                A_model_evals,
                B_model_evals,
                C_i_model_evals,
                D_i_model_evals,
                computed_indices["sobol_i"],
                scheme=second_order_scheme,
            )

            self.logger.info("UQpy: Second order Sobol indices computed successfully.")

            computed_indices["sobol_ij"] = self.sobol_ij

        ################## CONFIDENCE INTERVALS ####################

        if num_bootstrap_samples is not None:

            self.logger.info("UQpy: Computing confidence intervals ...")

            estimator_inputs = [
                A_model_evals,
                B_model_evals,
                C_i_model_evals,
                D_i_model_evals,
            ]

            # First order Sobol indices
            self.CI_sobol_i = self.bootstrapping(
                compute_first_order,
                estimator_inputs,
                computed_indices["sobol_i"],
                num_bootstrap_samples,
                confidence_level,
                scheme=first_order_scheme,
            )

            self.logger.info(
                "UQpy: Confidence intervals for First order Sobol indices computed successfully."
            )

            computed_indices["CI_sobol_i"] = self.CI_sobol_i

            # Total order Sobol indices
            self.CI_sobol_total_i = self.bootstrapping(
                compute_total_order,
                estimator_inputs,
                computed_indices["sobol_total_i"],
                num_bootstrap_samples,
                confidence_level,
                scheme=total_order_scheme,
            )

            self.logger.info(
                "UQpy: Confidence intervals for Total order Sobol indices computed successfully."
            )

            computed_indices["CI_sobol_total_i"] = self.CI_sobol_total_i

            # Second order Sobol indices
            if estimate_second_order:
                self.CI_sobol_ij = self.bootstrapping(
                    compute_second_order,
                    estimator_inputs,
                    computed_indices["sobol_ij"],
                    num_bootstrap_samples,
                    confidence_level,
                    first_order_sobol=computed_indices["sobol_i"],
                    scheme=second_order_scheme,
                )

                self.logger.info(
                    "UQpy: Confidence intervals for Second order Sobol indices computed successfully."
                )

                computed_indices["CI_sobol_ij"] = self.CI_sobol_ij

        return computed_indices


###################### Pick and Freeze Methods #####################

"""

These methods are also called by other sensitivity methods (such as Chatterjee, 
Cramer-von Mises) to estimate the Sobol indices and therefore are implemented as 
functions and not static methods in the Sobol class.


#! Saltelli2002
--------------------------------------------------------------------------------

Sobol indices estimated as per Theorem 2 in [3]_. Refer page 7 in
[3]_ for details.

Since there are several sets of function evaluations available,
there are several ways to estimate E[Y]^2 and V[Y].
Below we summarise the evaluations to be used as given in Theorem 2.

# First-order indices:
    - E[Y]^2 : f_A, f_B
    - V[Y] : f_A
    - S_i = ( <f_A, f_C_i>/N - E[Y]^2 ) / V[Y]


# Second-order indices:
    - Estimate 1:
        - E[Y]^2 : f_C_l, f_D_l -> l = max(i,j)
        - V[Y] : f_C_j or f_D_i
        - V^c_ij = f_D_i, f_C_j

    - Estimate 2:
        - E[Y]^2: f_C_l, f_D_l -> l = min(i,j)
        - V[Y] : f_C_i or f_D_j
        - V^c_ij = f_D_j, f_C_i

    where:
    S_ij = S^c_ij - S_i - S_j
    S^c_ij = ( <f_A, f_C_ij>/N - E[Y]^2 ) / V[Y] # Esimate 1
            = ( <f_B, f_D_ij>/N - E[Y]^2 ) / V[Y] # Esimate 2

# Total-order indices:
        - E[Y]^2 : f_B
        - V[Y] : f_B
        - S_T_i = 1 - ( <f_B, f_C_i>/N - E[Y]^2 ) / V[Y]

For m=5, the Sobol indices are estimated as follows:
First order indices: 2 estimates
Second order indices: 2 estimates
Total order indices: 2 estimates
S_{-ij}: 2 estimates
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
|       |   f_B  |  f_C_1  |  f_C_2  |  f_C_3  |  f_C_4  |  f_C_5 |  f_D_1  |  f_D_2  |  f_D_3  |  f_D_4  | f_D_5 |  f_A |
+=======+========+=========+=========+=========+=========+========+=========+=========+=========+=========+=======+======+
|  f_B  |  V[Y]  |         |         |         |         |        |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_C_1 |  S_T_1 |   V[Y]  |         |         |         |        |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_C_2 |  S_T_2 | V^c_-12 |   V[Y]  |         |         |        |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_C_3 |  S_T_3 | V^c_-13 | V^c_-23 |   V[Y]  |         |        |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_C_4 |  S_T_4 | V^c_-14 | V^c_-24 | V^c_-34 |   V[Y]  |        |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_C_5 |  S_T_5 | V^c_-15 | V^c_-25 | V^c_-35 | V^c_-45 |  V[Y]  |         |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_D_1 |   S_1  |  E^2[Y] |  V^c_12 |  V^c_13 |  V^c_14 | V^c_15 |   V[Y]  |         |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_D_2 |   S_2  |  V^c_12 |  E^2[Y] |  V^c_23 |  V^c_24 | V^c_25 | V^c_-12 |   V[Y]  |         |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_D_3 |   S_3  |  V^c_13 |  V^c_23 |  E^2[Y] |  V^c_34 | V^c_35 | V^c_-13 | V^c_-23 |   V[Y]  |         |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_D_4 |   S_4  |  V^c_14 |  V^c_24 |  V^c_34 |  E^2[Y] | V^c_45 | V^c_-14 | V^c_-24 | V^c_-34 |   V[Y]  |       |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
| f_D_5 |   S_5  |  V^c_15 |  V^c_25 |  V^c_35 |  V^c_45 | E^2[Y] | V^c_-15 | V^c_-25 | V^c_-35 | V^c_-45 |  V[Y] |      |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+
|  f_A  | E^2[Y] |   S_1   |   S_2   |   S_3   |   S_4   |   S_5  |  S_T_1  |  S_T_2  |  S_T_3  |  S_T_4  | S_T_5 | V[Y] |
+-------+--------+---------+---------+---------+---------+--------+---------+---------+---------+---------+-------+------+

For m>5, we can follow the same procedure as above.

For m = 4, the Sobol indices are estimated as follows:
First order indices: 2 estimates
Second order indices: 4 estimates
Total order indices: 2 estimates
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
|       |   f_B  |  f_C_1 |  f_C_2 |  f_C_3 |  f_C_4 |  f_D_1 |  f_D_2 |  f_D_3 | f_D_4 |  f_A |
+=======+========+========+========+========+========+========+========+========+=======+======+
|  f_B  |  V[Y]  |        |        |        |        |        |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_C_1 |  S_T_1 |  V[Y]  |        |        |        |        |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_C_2 |  S_T_2 | V^c_34 |  V[Y]  |        |        |        |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_C_3 |  S_T_3 | V^c_24 | V^c_14 |  V[Y]  |        |        |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_C_4 |  S_T_4 | V^c_23 | V^c_13 | V^c_12 |  V[Y]  |        |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_D_1 |   S_1  | E^2[Y] | V^c_12 | V^c_13 | V^c_14 |  V[Y]  |        |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_D_2 |   S_2  | V^c_12 | E^2[Y] | V^c_23 | V^c_24 | V^c_34 |  V[Y]  |        |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_D_3 |   S_3  | V^c_13 | V^c_23 | E^2[Y] | V^c_34 | V^c_25 | V^c_14 |  V[Y]  |       |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
| f_D_4 |   S_4  | V^c_14 | V^c_24 | V^c_34 | E^2[Y] | V^c_23 | V^c_13 | V^c_12 |  V[Y] |      |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+
|  f_A  | E^2[Y] |   S_1  |   S_2  |   S_3  |   S_4  |  S_T_1 |  S_T_2 |  S_T_3 | S_T_4 | V[Y] |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+------+

For m = 3, the Sobol indices are estimated as follows:
First order indices: 4 estimates
Second order indices: 2 estimates
Total order indices: 2 estimates
+-------+--------+--------+--------+--------+-------+-------+-------+------+
|       |   f_B  |  f_C_1 |  f_C_2 |  f_C_3 | f_D_1 | f_D_2 | f_D_3 |  f_A |
+=======+========+========+========+========+=======+=======+=======+======+
|  f_B  |  V[Y]  |        |        |        |       |       |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_C_1 |  S_T_1 |  V[Y]  |        |        |       |       |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_C_2 |  S_T_2 |   S_3  |  V[Y]  |        |       |       |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_C_3 |  S_T_3 |   S_2  |   S_1  |  V[Y]  |       |       |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_D_1 |   S_1  | E^2[Y] | V^c_12 | V^c_13 |  V[Y] |       |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_D_2 |   S_2  | V^c_12 | E^2[Y] | V^c_23 |  S_3  |  V[Y] |       |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
| f_D_3 |   S_3  | V^c_13 | V^c_23 | E^2[Y] |  S_2  |  S_1  |  V[Y] |      |
+-------+--------+--------+--------+--------+-------+-------+-------+------+
|  f_A  | E^2[Y] |   S_1  |   S_2  |   S_3  | S_T_1 | S_T_2 | S_T_3 | V[Y] |
+-------+--------+--------+--------+--------+-------+-------+-------+------+

"""


def compute_first_order(
    A_model_evals,
    B_model_evals,
    C_i_model_evals,
    D_i_model_evals=None,
    scheme="Janon2014",
):

    """
    Compute first order Sobol' indices using the Pick-and-Freeze scheme.

    For the Sobol1996 scheme:
    For computing the first order Sobol' indices, only f_A_model_evals and
    f_C_i_model_evals are required. The other inputs are optional.
        f_B_model_evals is set to None if f_B_model_evals is not provided.

    **Inputs:**

    * **A_model_evals** (`ndarray`):
        Shape: `(n_samples, n_outputs)`.

    * **B_model_evals** (`ndarray`):
        If not available, pass `None`.
        Shape: `(n_samples, n_outputs)`.

    * **C_i_model_evals** (`ndarray`):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **D_i_model_evals** (`ndarray`, optional):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **scheme** (`str`, optional):
        Scheme to use for computing the first order Sobol' indices.
        Default: 'Sobol1993'.

    **Outputs:**

    * **first_order_sobol** (`ndarray`):
        First order Sobol' indices.
        Shape: `(num_vars, n_outputs)`.

    """

    n_samples = A_model_evals.shape[0]
    n_outputs = A_model_evals.shape[1]
    num_vars = C_i_model_evals.shape[2]

    # Store first order Sobol' indices
    first_order_sobol = np.zeros((num_vars, n_outputs))

    if scheme == "Sobol1993":

        for output_j in range(n_outputs):

            f_A = A_model_evals[:, output_j]
            f_B = B_model_evals[:, output_j] if B_model_evals is not None else None

            # combine all model evaluations
            # to improve accuracy of the estimator
            _all_model_evals = np.append(f_A, f_B) if f_B is not None else f_A
            f_0 = np.mean(_all_model_evals)  # scalar

            f_0_square = f_0**2
            total_variance = np.var(_all_model_evals, ddof=1)

            for var_i in range(num_vars):

                f_C_i = C_i_model_evals[output_j, :, var_i]

                S_i = (np.dot(f_A, f_C_i) / n_samples - f_0_square) / total_variance

                first_order_sobol[var_i, output_j] = S_i

    elif scheme == "Janon2014":

        for output_j in range(n_outputs):

            f_A = A_model_evals[:, output_j]

            for var_i in range(num_vars):

                f_C_i = C_i_model_evals[output_j, :, var_i]

                # combine all model evaluations
                # to improve accuracy of the estimator
                _all_model_evals = np.append(f_A, f_C_i)
                f_0 = np.mean(_all_model_evals)

                f_0_square = f_0**2
                total_variance = np.mean(_all_model_evals**2) - f_0_square

                S_i = (np.dot(f_A, f_C_i) / n_samples - f_0_square) / total_variance

                first_order_sobol[var_i, output_j] = S_i

    elif scheme == "Saltelli2002":

        """
        Number of estimates for first order indices is 4 if
        num_vars is 3, else 2.

        """

        for output_j in range(n_outputs):

            f_A = A_model_evals[:, output_j]
            f_B = B_model_evals[:, output_j]
            f_0_square = np.dot(f_A, f_B) / n_samples
            total_variance = np.var(f_A, ddof=1)

            for var_i in range(num_vars):

                f_C_i = C_i_model_evals[output_j, :, var_i]
                f_D_i = D_i_model_evals[output_j, :, var_i]

                # (Estimate 1)
                est_1 = (np.dot(f_A, f_C_i) / n_samples - f_0_square) / total_variance

                # (Estimate 2)
                est_2 = (np.dot(f_B, f_D_i) / n_samples - f_0_square) / total_variance

                if num_vars == 3:

                    # list of variable indices
                    list_vars = list(range(num_vars))
                    list_vars.remove(var_i)
                    # combination of all remaining variables indices
                    rem_vars_perm = list(itertools.permutations(list_vars, 2))

                    # (Estimate 3)
                    var_a, var_b = rem_vars_perm[0]
                    f_C_a = C_i_model_evals[output_j, :, var_a]
                    f_C_b = C_i_model_evals[output_j, :, var_b]
                    est_3 = (
                        np.dot(f_C_a, f_C_b) / n_samples - f_0_square
                    ) / total_variance

                    # (Estimate 4)
                    var_a, var_b = rem_vars_perm[1]
                    f_D_a = D_i_model_evals[output_j, :, var_a]
                    f_D_b = D_i_model_evals[output_j, :, var_b]
                    est_4 = (
                        np.dot(f_D_a, f_D_b) / n_samples - f_0_square
                    ) / total_variance

                    first_order_sobol[var_i, output_j] = (
                        est_1 + est_2 + est_3 + est_4
                    ) / 4

                else:
                    first_order_sobol[var_i, output_j] = (est_1 + est_2) / 2

    return first_order_sobol


def compute_total_order(
    A_model_evals,
    B_model_evals,
    C_i_model_evals,
    D_i_model_evals=None,
    scheme="Homma1996",
):

    """
    Compute total order Sobol' indices using the Pick-and-Freeze scheme.

    For the Homma1996 scheme:
    For computing the first order Sobol' indices, only f_B_model_evals and
    f_C_i_model_evals are required.
        f_A_model_evals is set to None if f_A_model_evals is not provided.

    **Inputs:**

    * **A_model_evals** (`ndarray`):
        If not available, pass `None`.
        Shape: `(n_samples, n_outputs)`.

    * **B_model_evals** (`ndarray`):
        Shape: `(n_samples, n_outputs)`.

    * **C_i_model_evals** (`ndarray`):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **D_i_model_evals** (`ndarray`, optional):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **scheme** (`str`, optional):
        Scheme to use for computing the total order Sobol' indices.
        Default: 'Homma1996'.

    **Outputs:**

    * **total_order_sobol** (`ndarray`):
        Total order Sobol' indices.
        Shape: `(num_vars, n_outputs)`.

    """

    n_samples = A_model_evals.shape[0]
    n_outputs = A_model_evals.shape[1]
    num_vars = C_i_model_evals.shape[2]

    # Store total order Sobol' indices
    total_order_sobol = np.zeros((num_vars, n_outputs))

    if scheme == "Homma1996":

        for output_j in range(n_outputs):

            f_A = A_model_evals[:, output_j] if A_model_evals is not None else None
            f_B = B_model_evals[:, output_j]

            # combine all model evaluations
            # to improve accuracy of the estimator
            _all_model_evals = np.append(f_A, f_B) if f_A is not None else f_B
            f_0 = np.mean(_all_model_evals)  # scalar

            f_0_square = f_0**2
            total_variance = np.var(_all_model_evals, ddof=1)

            for var_i in range(num_vars):

                f_C_i = C_i_model_evals[output_j, :, var_i]

                S_T_i = (
                    1 - (np.dot(f_B, f_C_i) / n_samples - f_0_square) / total_variance
                )

                total_order_sobol[var_i, output_j] = S_T_i

    elif scheme == "Saltelli2002":

        for output_j in range(n_outputs):

            f_A = A_model_evals[:, output_j]
            f_B = B_model_evals[:, output_j]
            f_0_square = np.mean(f_B) ** 2
            total_variance = np.var(f_B, ddof=1)

            for var_i in range(num_vars):

                f_C_i = C_i_model_evals[output_j, :, var_i]
                f_D_i = D_i_model_evals[output_j, :, var_i]

                # (Estimate 1)
                est_1 = (
                    1 - (np.dot(f_B, f_C_i) / n_samples - f_0_square) / total_variance
                )

                # (Estimate 2)
                est_2 = (
                    1 - (np.dot(f_A, f_D_i) / n_samples - f_0_square) / total_variance
                )

                total_order_sobol[var_i, output_j] = (est_1 + est_2) / 2

    return total_order_sobol


def compute_second_order(
    A_model_evals,
    B_model_evals,
    C_i_model_evals,
    D_i_model_evals,
    first_order_sobol=None,  # None to make it a make keyword argument
    scheme="Saltelli2002",
):
    """
    Compute the second order Sobol indices using the Pick-and-Freeze scheme.

    NOTE:
    - Number of estimates for second order indices is 4 if
        num_vars is 4, else 2.

    - Although the B_model_evals are not being used currently, they are
        included for use in estimate 3 and 4 for case num_vars = 4.

    **Inputs:**

    * **A_model_evals** (`ndarray`):
        Shape: `(n_samples, n_outputs)`.

    * **B_model_evals** (`ndarray`):
        Shape: `(n_samples, n_outputs)`.

    * **C_i_model_evals** (`ndarray`):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **D_i_model_evals** (`ndarray`, optional):
        Shape: `(n_outputs, n_samples, num_vars)`.

    * **first_order_sobol** (`ndarray`):
        First order Sobol' indices.
        Shape: `(num_vars, n_outputs)`.

    * **scheme** (`str`, optional):
        Scheme to use for computing the first order Sobol' indices.
        Default: 'Sobol1993'.

    **Outputs:**

    * **second_order_sobol** (`ndarray`):
        Second order Sobol indices.
        Shape: `(num_second_order_terms, n_outputs)`.
    """

    n_samples = A_model_evals.shape[0]
    n_outputs = A_model_evals.shape[1]
    num_vars = C_i_model_evals.shape[2]

    second_order_terms = itertools.combinations(range(num_vars), 2)
    second_order_terms = list(second_order_terms)
    num_second_order_terms = math.comb(num_vars, 2)

    # Store second order Sobol' indices
    second_order_sobol = np.zeros((num_second_order_terms, n_outputs))

    if scheme == "Saltelli2002":

        for output_j in range(n_outputs):

            for k in range(num_second_order_terms):

                var_a, var_b = second_order_terms[k]
                S_a = first_order_sobol[var_a, output_j]
                S_b = first_order_sobol[var_b, output_j]

                # (Estimate 1)
                var_c = np.max([var_a, var_b])
                f_C_c = C_i_model_evals[output_j, :, var_c]
                f_D_c = D_i_model_evals[output_j, :, var_c]
                f_0_square = np.dot(f_D_c, f_C_c) / n_samples
                total_variance = np.var(f_D_c, ddof=1)

                f_C_a = C_i_model_evals[output_j, :, var_a]
                f_D_b = D_i_model_evals[output_j, :, var_b]
                S_c_ab_1 = (
                    np.dot(f_C_a, f_D_b) / n_samples - f_0_square
                ) / total_variance

                est_1 = S_c_ab_1 - S_a - S_b

                # (Estimate 2)
                var_c = np.min([var_a, var_b])
                f_C_c = C_i_model_evals[output_j, :, var_c]
                f_D_c = D_i_model_evals[output_j, :, var_c]
                f_0_square = np.dot(f_D_c, f_C_c) / n_samples
                total_variance = np.var(f_D_c, ddof=1)

                f_D_a = D_i_model_evals[output_j, :, var_a]
                f_C_b = C_i_model_evals[output_j, :, var_b]
                S_c_ab_2 = (
                    np.dot(f_D_a, f_C_b) / n_samples - f_0_square
                ) / total_variance

                est_2 = S_c_ab_2 - S_a - S_b

                if num_vars == 4:

                    # (Estimate 3)
                    # TODO: How to compute this?

                    # (Estimate 4)
                    # TODO: How to compute this?

                    # second_order_sobol[k, output_j] = (
                    #     est_1 + est_2 + est_3 + est_4
                    # ) / 4

                    pass

                else:
                    second_order_sobol[k, output_j] = (est_1 + est_2) / 2

    return second_order_sobol
