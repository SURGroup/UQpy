"""
This module contains the Chatterjee coefficient of correlation proposed 
in [1]_. 

Using the rank statistics, we can also estimate the Sobol indices proposed by 
Gamboa et al. [2]_.

References
----------

.. [1] Sourav Chatterjee (2021) A New Coefficient of Correlation, Journal of the
        American Statistical Association, 116:536, 2009-2022, 
        DOI: 10.1080/01621459.2020.1758115

.. [2] Fabrice Gamboa, Pierre Gremaud, Thierry Klein, and Agnès Lagnoux. (2020). 
        Global Sensitivity Analysis: a new generation of mighty estimators 
        based on rank statistics.

"""

import logging

import numpy as np
import scipy.stats
from beartype import beartype
from typing import Union
from numbers import Integral

from UQpy.sensitivity.baseclass.Sensitivity import Sensitivity
from UQpy.sensitivity.SobolSensitivity import compute_first_order as compute_first_order_sobol
from UQpy.utilities.ValidationTypes import (
    RandomStateType,
    PositiveInteger,
    PositiveFloat,
    NumpyFloatArray,
    NumpyIntArray,
)
from UQpy.utilities.UQpyLoggingFormatter import UQpyLoggingFormatter


class ChatterjeeSensitivity(Sensitivity):
    """
    Compute sensitivity indices using the Chatterjee correlation coefficient.

    Using the same model evaluations, we can also estimate the Sobol indices.

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

    def __init__(self, runmodel_object, dist_object, random_state=None):
        super().__init__(runmodel_object, dist_object, random_state=random_state)

        # Create logger with the same name as the class
        self.logger = logging.getLogger(__name__)

        self.first_order_chatterjee_indices = None
        "Chatterjee sensitivity indices (First order), :class:`numpy.ndarray` of shape :code:`(n_variables, 1)`"

        self.first_order_sobol_indices = None
        "Sobol indices computed using the rank statistics, :class:`numpy.ndarray` of shape :code:`(n_variables, 1)`"

        self.confidence_interval_chatterjee = None
        "Confidence intervals for the Chatterjee sensitivity indices, :class:`numpy.ndarray` of " \
        "shape :code:`(n_variables, 2)`"

        self.n_variables = None
        "Number of input random variables, :class:`int`"

        self.n_samples = None
        "Number of samples used to estimate the sensitivity indices, :class:`int`"

    @beartype
    def run(
        self,
        n_samples: PositiveInteger = 1_000,
        estimate_sobol_indices: bool = False,
        n_bootstrap_samples: PositiveInteger = None,
        confidence_level: PositiveFloat = 0.95,
    ):
        """
        Compute the sensitivity indices using the Chatterjee method. Employing the :code:`run` method will initialize
        :code:`n_samples` simulations using :class:`.RunModel`. To compute sensitivity indices using pre-computed inputs
        and outputs, use the static methods described below.

        :param n_samples: Number of samples used to compute the Chatterjee indices. \
            Default is 1,000.   

        :param estimate_sobol_indices: If :code:`True`, the Sobol indices are estimated \
            using the pick-and-freeze samples.

        :param n_bootstrap_samples: Number of bootstrap samples used to estimate the \
            Sobol indices. Default is :any:`None`.

        :param confidence_level: Confidence level used to compute the confidence \
            intervals of the Cramér-von Mises indices.
        """

        # Check nsamples
        self.n_samples = n_samples
        if not isinstance(self.n_samples, int):
            raise TypeError("UQpy: nsamples should be an integer")

        # Check num_bootstrap_samples data type
        if n_bootstrap_samples is None:
            self.logger.info(
                "UQpy: num_bootstrap_samples is set to None, confidence intervals will not be computed.\n")
        elif not isinstance(n_bootstrap_samples, int):
            raise TypeError("UQpy: num_bootstrap_samples should be an integer.\n")

        ################## GENERATE SAMPLES ##################

        A_samples = self.dist_object.rvs(self.n_samples, random_state=self.random_state)

        self.logger.info("UQpy: Generated samples successfully.\n")

        self.n_variables = A_samples.shape[1]  # number of variables

        ################# MODEL EVALUATIONS ####################

        A_model_evals = self._run_model(A_samples).reshape(-1, 1)

        self.logger.info("UQpy: Model evaluations completed.\n")


        ################## COMPUTE CHATTERJEE INDICES ##################

        self.first_order_chatterjee_indices = self.compute_chatterjee_indices(A_samples, A_model_evals)

        self.logger.info("UQpy: Chatterjee indices computed successfully.\n")


        ################## COMPUTE SOBOL INDICES ##################

        self.logger.info("UQpy: Computing First order Sobol indices ...\n")

        if estimate_sobol_indices:
            f_C_i_model_evals = self.compute_rank_analog_of_f_C_i(A_samples, A_model_evals)

            self.first_order_sobol_indices = self.compute_Sobol_indices(A_model_evals, f_C_i_model_evals)

            self.logger.info("UQpy: First order Sobol indices computed successfully.\n")


        ################## CONFIDENCE INTERVALS ####################

        if n_bootstrap_samples is not None:

            self.logger.info("UQpy: Computing confidence intervals ...\n")

            estimator_inputs = [A_samples, A_model_evals]

            self.confidence_interval_chatterjee = self.bootstrapping(
                self.compute_chatterjee_indices,
                estimator_inputs,
                self.first_order_chatterjee_indices,
                n_bootstrap_samples,
                confidence_level,
            )

            self.logger.info("UQpy: Confidence intervals for Chatterjee indices computed successfully.\n")


    @staticmethod
    @beartype
    def compute_chatterjee_indices(
        X: Union[NumpyFloatArray, NumpyIntArray],
        Y: Union[NumpyFloatArray, NumpyIntArray],
        seed: RandomStateType = None,
    ):
        r"""

        Compute the Chatterjee sensitivity indices
        between the input random vectors :math:`X=\left[ X_{1}, X_{2},…,X_{d} \right]`
        and output random vector Y.

        :param X: Input random vectors, :class:`numpy.ndarray` of shape :code:`(n_samples, n_variables)`

        :param Y: Output random vector, :class:`numpy.ndarray` of shape :code:`(n_samples, 1)`

        :param seed: Seed for the random number generator.

        :return: Chatterjee sensitivity indices, :class:`numpy.ndarray` of shape :code:`(n_variables, 1)`

        """

        if seed is not None:
            # set seed for reproducibility
            np.random.seed(seed)

        N = X.shape[0]  # number of samples
        m = X.shape[1]  # number of variables

        chatterjee_indices = np.zeros((m, 1))

        for i in range(m):

            # Samples of random variable X_i
            X_i = X[:, i].reshape(-1, 1)

            #! For ties in X_i
            # we break ties uniformly at random
            # Shuffle X_i and Y
            _ix = np.arange(N)  # indices of X_i
            np.random.shuffle(_ix)  # shuffle indices
            X_i_shuffled = X_i[_ix]  # shuffle X_i
            Y_shuffled = Y[_ix]  # shuffle Y

            Z = np.hstack((X_i_shuffled, Y_shuffled))

            # Sort the columns of Z by X_i
            # such that the tuple (X_i, Y_i) is unchanged
            Z_sorted = Z[Z[:, 0].argsort()]

            # Find rank of y_i in the sorted columns of Y
            # r[i] is number of j s.t. y[j] <= y[i],
            # This is accomplished using rankdata with method='max'
            # Example: Y = [1, 2, 3, 3, 4, 5], rank = [1, 2, 4, 4, 5, 6]
            rank = scipy.stats.rankdata(Z_sorted[:, 1], method="max")

            #! For ties in Y
            # l[i] is number of j s.t. y[i] <= y[j],
            # This is accomplished using rankdata with method='max'
            # Example: Y = [1, 2, 3, 3, 4, 5], l = [6, 5, 4, 4, 2, 1]
            # One could also use the Y_shuffled array, since sum2 only
            # multiplies terms of same index, i.e l_i*(n - l_i)
            L = scipy.stats.rankdata(-Z_sorted[:, 1], method="max")

            sum1 = np.abs(rank[1:] - rank[:-1]).sum()

            sum2 = np.sum(L * (N - L))

            chatterjee_indices[i] = 1 - N * sum1 / (2 * sum2)

        return chatterjee_indices

    @staticmethod
    @beartype
    def rank_analog_to_pickfreeze(
        X: Union[NumpyFloatArray, NumpyIntArray], j: Integral
    ):
        r"""
        Computing the :math:`N(j)` for each :math:`j \in \{1, \ldots, n\}`
        as in eq.(8) in :cite:`gamboa2020global`, where :math:`n` is the size of :math:`X`.

        .. math::
            :nowrap:

                \begin{equation}
                    N(j):=
                    \begin{cases}
                        \pi^{-1}(\pi(j)+1) &\text { if } \pi(j)+1 \leqslant n \\
                        \pi^{-1}(1) &\text { if } \pi(j)=n
                    \end{cases}
                \end{equation}

        where, :math:`\pi(j) := \mathrm{rank}(x_j)`

        :param X: Input random vector, :class:`numpy.ndarray` of shape :code:`(n_samples, 1)`

        :param j: Index of the sample :math:`j \in \{1, \ldots, n\}`

        :return: :math:`N(j)` :class:`int`

        """

        N = X.shape[0]  # number of samples

        # Ranks of elements of X_i
        # -1 so that the ranks are 0-based
        # for convenience in indexing
        rank_X = scipy.stats.rankdata(X) - 1
        rank_X = rank_X.astype(int)

        # Find rank of element j
        rank_j = rank_X[j]

        if rank_j + 1 <= N - 1:
            # Get index of element: rank_j + 1
            return np.where(rank_X == rank_j + 1)[0][0]

        if rank_j == N - 1:
            return np.where(rank_X == 0)[0][0]

    @staticmethod
    @beartype
    def rank_analog_to_pickfreeze_vec(X: Union[NumpyFloatArray, NumpyIntArray]):
        r"""
        Computing the :math:`N(j)` for each :math:`j \in \{1, \ldots, n\}`
        in a vectorized manner., where :math:`n` is the size of :math:`X`.

        This method is significantly faster than the looping version
        ``rank_analog_to_pickfreeze`` but is also more complicated.

        .. math::
            :nowrap:

                \begin{equation}
                    N(j):=
                    \begin{cases}
                        \pi^{-1}(\pi(j)+1) &\text { if } \pi(j)+1 \leqslant n \\
                        \pi^{-1}(1) &\text { if } \pi(j)=n
                    \end{cases}
                \end{equation}
        
        where, :math:`\pi(j) := \mathrm{rank}(x_j)`

        Key idea: :math:`\pi^{-1}` is rank_X.argsort() (
        `see also <https://rdrr.io/cran/sensitivity/src/R/sobolrank.R>`_)

        Example:
        X = [22, 74, 44, 11, 1]

        N_J = [3, 5, 2, 1, 4] (1-based indexing)

        N_J = [2, 4, 1, 0, 3] (0-based indexing)

        :param X: Input random vector, :class:`numpy.ndarray` of shape :code:`(n_samples, 1)`

        :return: :math:`N(j)`, :class:`numpy.ndarray` of shape :code:`(n_samples, 1)`

        """

        N = X.shape[0]  # number of samples
        N_func = np.zeros((N, 1))

        # Ranks of elements of X_i
        # -1 since ranks are 0-based
        rank_X = scipy.stats.rankdata(X, method="ordinal") - 1
        rank_X = rank_X.astype(int)

        # Inverse of pi(j): j = pi^-1(rank_X(j))
        #! This is non-trivial
        pi_inverse = rank_X.argsort()  # complexity: N*log(N)

        # CONDITION 2
        # Find j with rank_j == N-1
        j_meets_condition_2 = pi_inverse[N - 1]
        N_func[j_meets_condition_2] = pi_inverse[0]

        # CONDITION 1
        # Find j's with rank_j + 1 <= N-1
        # term_1 = pi(j) + 1
        j_remaining = np.delete(np.arange(N), j_meets_condition_2)
        term_1 = rank_X[j_remaining] + 1

        j_remaining_meet_condition_1 = pi_inverse[term_1]

        # j_remaining_meet_condition_1 = np.where(rank_X_i == condition)
        N_func[j_remaining, 0] = j_remaining_meet_condition_1

        return N_func.astype(int)

    @staticmethod
    @beartype
    def compute_Sobol_indices(
        A_model_evals: Union[NumpyFloatArray, NumpyIntArray],
        C_i_model_evals: Union[NumpyFloatArray, NumpyIntArray],
    ):
        r"""
        A method to estimate the first order Sobol indices using
        the Chatterjee method.

        .. math::
            :nowrap:

            \begin{equation}
                \xi_{n}^{\mathrm{Sobol}}\left(X_{1}, Y\right):=
                \frac{\frac{1}{n} \sum_{j=1}^{n} Y_{j} Y_{N(j)}-\left(\frac{1}{n} \sum_{j=1}^{n} Y_{j}\right)^{2}}
                {\frac{1}{n} \sum_{j=1}^{n}\left(Y_{j}\right)^{2}-\left(\frac{1}{n} \sum_{j=1}^{n} Y_{j}\right)^{2}}
            \end{equation}

        where the term :math:`Y_{N(j)}` is computed using the method:``rank_analog_to_pickfreeze_vec``.

        :param A_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, 1)`

        :param C_i_model_evals: Model evaluations, :class:`numpy.ndarray` of shape :code:`(n_samples, n_variables)`

        :return: First order Sobol indices, :class:`numpy.ndarray` of shape :code:`(n_variables, 1)`

        """

        # extract shape
        _shape = C_i_model_evals.shape

        # convert C_i_model_evals to 3D array
        # with n_outputs=1 in first dimension
        n_outputs = 1
        C_i_model_evals = C_i_model_evals.reshape((n_outputs, *_shape))

        first_order_sobol = compute_first_order_sobol(A_model_evals, None, C_i_model_evals, scheme="Sobol1993")

        return first_order_sobol

    @beartype
    def compute_rank_analog_of_f_C_i(
        self,
        A_samples: Union[NumpyFloatArray, NumpyIntArray],
        A_model_evals: Union[NumpyFloatArray, NumpyIntArray],
    ):
        r"""
        In the Pick and Freeze method, we use model evaluations
        :math:`f_A`, :math:`f_B`, :math:`f_{C_{i}}`
        to compute the Sobol indices.

        Gamboa et al. provide a rank analog to :math:`f_{C_{i}}` in eq. (6) in [6]_.

        **Inputs:**

        * **A_samples** (`ndarray`):
            Shape: `(n_samples, n_variables)`.

        * **A_model_evals** (`ndarray`):
            Shape: `(n_samples, 1)`.

        **Outputs:**

        * **A_i_model_evals** (`ndarray`):
            Shape: `(n_samples, n_variables)`.

        """

        f_A = A_model_evals
        N = f_A.shape[0]
        m = self.n_variables

        A_i_model_evals = np.zeros((N, m))

        for i in range(m):

            K = self.rank_analog_to_pickfreeze_vec(A_samples[:, i])

            A_i_model_evals[:, i] = f_A[K].ravel()

        return A_i_model_evals
