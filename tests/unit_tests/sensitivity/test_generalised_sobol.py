""""
This is the test module for the Generalised Sobol indices.

Here, we will use the toy example from [1]_, which is a multi-output problem.


References
----------

.. [1]  Gamboa F, Janon A, Klein T, Lagnoux A, others. 
        Sensitivity analysis for multidimensional and functional outputs.
        Electronic journal of statistics 2014; 8(1): 575-603.

Important
----------
The computed indices are computed using the `np.isclose` function.

Function signature:
    numpy.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

    Parameters:
    a, b: array_like
        Input arrays to compare.

    rtol: float
        The relative tolerance parameter.

    atol: float
        The absolute tolerance parameter.

Each element of the `diff` array is compared as follows:
diff = |a - b|
diff <= atol + rtol * abs(b)

-   relative tolerance: rtol * abs(b)
    It is the maximum allowed difference between a and b, 
    relative to the absolute value of b. 
    For example, to set a tolerance of 1%, pass rol=0.01, 
    which assures that the values are within 2 decimal places of each other.
-   absolute tolerance: atol
    When b is close to zero, the atol value is used.

"""

import numpy as np
import pytest
import scipy

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform, Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.GeneralisedSobolSensitivity import GeneralisedSobolSensitivity

# Prepare
###############################################################################

# Prepare the input distribution
@pytest.fixture()
def normal_input_dist_object():
    """
    This function returns the input distribution for the toy model.

    X1 ~ Normal(0, 1)
    X2 ~ Normal(0, 1)

    """
    return JointIndependent([Normal(0, 1)] * 2)


@pytest.fixture()
def uniform_input_dist_object():
    """
    This function returns the input distribution for the toy model.

    X1 ~ Uniform(0, 1)
    X2 ~ Uniform(0, 1)

    """
    return JointIndependent([Uniform(0, 1)] * 2)


@pytest.fixture()
def toy_model_object():
    """
    This function creates the toy model.

    """
    model = PythonModel(
        model_script="multioutput.py",
        model_object_name="evaluate",
        var_names=[
            "X_1",
            "X_2",
        ],
        delete_files=True,
    )

    runmodel_obj = RunModel(model=model)

    return runmodel_obj


@pytest.fixture()
def generalised_sobol_object_normal(normal_input_dist_object, toy_model_object):
    """
    This function creates the Generalised Sobol indices object
    with normal input distribution.

    """

    return GeneralisedSobolSensitivity(toy_model_object, normal_input_dist_object)


@pytest.fixture()
def generalised_sobol_object_uniform(uniform_input_dist_object, toy_model_object):
    """
    This function creates the Generalised Sobol indices object
    with uniform input distribution.

    """

    return GeneralisedSobolSensitivity(toy_model_object, uniform_input_dist_object)


@pytest.fixture()
def analytical_toy_GSI_normal():
    """
    Analytical first order Generalised Sobol indices
    for the toy example with normal input distribution.
    """

    return np.array([0.2941, 0.1176]).reshape(-1, 1)


@pytest.fixture()
def analytical_toy_GSI_uniform():
    """ "
    Analytical first order Generalised Sobol indices
    for toy example with uniform input distribution.
    """

    return np.array([0.6084, 0.3566]).reshape(-1, 1)


@pytest.fixture()
def pick_and_freeze_toy_GSI_normal(generalised_sobol_object_normal):
    """ "
    Generalised first order Sobol indices computed using the Pick and Freeze
    approach for the toy example with normal input distribution.
    """

    SA = generalised_sobol_object_normal

    np.random.seed(12345)  #! set seed for reproducibility

    computed_indices = SA.run(n_samples=100_000)

    return computed_indices["gen_sobol_i"]


@pytest.fixture()
def pick_and_freeze_toy_GSI_uniform(generalised_sobol_object_uniform):
    """ "
    Generalised first order Sobol indices computed using the Pick and Freeze
    approach for the toy example with uniform input distribution.
    """

    SA = generalised_sobol_object_uniform

    np.random.seed(12345)  #! set seed for reproducibility

    computed_indices = SA.run(n_samples=100_000)

    return computed_indices["gen_sobol_i"]


@pytest.fixture()
def NUM_SAMPLES():
    """This function returns the number of samples for bootstrapping"""

    num_bootstrap_samples = 500
    num_samples = 20_000

    return num_bootstrap_samples, num_samples


@pytest.fixture()
def bootstrap_generalised_sobol_index_variance(
    generalised_sobol_object_normal, NUM_SAMPLES
):

    SA = generalised_sobol_object_normal

    np.random.seed(12345)  #! set seed for reproducibility

    num_bootstrap_samples, n_samples = NUM_SAMPLES

    confidence_level = 0.95
    delta = -scipy.stats.norm.ppf((1 - confidence_level) / 2)

    # Compute the confidence intervals

    computed_indices = SA.run(
        n_samples=n_samples,
        num_bootstrap_samples=num_bootstrap_samples,
        confidence_level=confidence_level,
    )

    gen_sobol_i = computed_indices["gen_sobol_i"].ravel()
    gen_sobol_total_i = computed_indices["gen_sobol_total_i"].ravel()
    upper_bound_first_order = computed_indices["confidence_interval_gen_sobol_i"][:, 1]
    upper_bound_total_order = computed_indices["confidence_interval_gen_sobol_total_i"][
        :, 1
    ]

    std_bootstrap_first_order = (upper_bound_first_order - gen_sobol_i) / delta
    std_bootstrap_total_order = (upper_bound_total_order - gen_sobol_total_i) / delta

    return std_bootstrap_first_order**2, std_bootstrap_total_order**2


@pytest.fixture()
def model_eval_generalised_sobol_index_variance():

    """
    For computational efficiency, the variance of the generalised Sobol indices
    is precomputed using model evaluations with
    NUM_SAMPLES (num_repetitions=500, num_samples=20_000)

    Copy-paste the following code to generate the variance
    of the Sobol indices:

    runmodel_obj = RunModel(model_script='multioutput.py',
                        model_object_name='multioutput_toy',
                        vec=True, delete_files=True)

    dist_object_1 = JointInd([Normal(0, 1)]*2)

    SA = GeneralisedSobol(runmodel_obj, dist_object_1)

    np.random.seed(12345) # for reproducibility

    num_repetitions, n_samples = 500, 20_000

    num_vars = 2

    bootstrap_first_order = np.zeros((num_vars, num_bootstrap_samples))
    bootstrap_total_order = np.zeros((num_vars, num_bootstrap_samples))

    for b in range(num_repetitions):

        computed_indices = SA.run(n_samples=n_samples)

        bootstrap_first_order[:, b] = computed_indices["gen_sobol_i"].ravel()
        bootstrap_total_order[:, b] = computed_indices["gen_sobol_total_i"].ravel()

    var_bootstrap_gen_S = np.var(bootstrap_first_order, axis=1, ddof=1)
    var_bootstrap_gen_S_T = np.var(bootstrap_total_order, axis=1, ddof=1)

    print(var_bootstrap_gen_S)
    print(var_bootstrap_gen_S_T)

    """

    variance_first_order = np.array([0.00011284, 0.00012608])

    variance_total_order = np.array([0.00012448, 0.00011208])

    return variance_first_order, variance_total_order


# Unit tests
###############################################################################


def test_pick_and_freeze_estimator(
    pick_and_freeze_toy_GSI_normal,
    analytical_toy_GSI_normal,
    pick_and_freeze_toy_GSI_uniform,
    analytical_toy_GSI_uniform,
):
    """
    Test the pick and freeze estimator.

    """

    # Prepare
    N_true = analytical_toy_GSI_normal
    N_estimate = pick_and_freeze_toy_GSI_normal

    U_true = analytical_toy_GSI_uniform
    U_estimate = pick_and_freeze_toy_GSI_uniform

    # Act
    # Idea: Measure accuracy upto 2 decimal places -> rtol=0, atol=1e-2
    assert np.isclose(N_estimate, N_true, rtol=0, atol=1e-2).all()
    assert np.isclose(U_estimate, U_true, rtol=0, atol=1e-2).all()


def test_bootstrap_variance_computation(
    model_eval_generalised_sobol_index_variance,
    bootstrap_generalised_sobol_index_variance,
):

    """Test the bootstrap variance computation."""

    # Prepare
    var_first, var_total = model_eval_generalised_sobol_index_variance
    boot_var_first, boot_var_total = bootstrap_generalised_sobol_index_variance

    # Act
    assert var_first.shape == boot_var_first.shape

    # Idea: Ensure bootstrap variance and MC variance are of same order -> rtol=0, atol=1e-4
    assert np.isclose(boot_var_first, var_first, rtol=0, atol=1e-4).all()
    assert np.isclose(boot_var_total, var_total, rtol=0, atol=1e-4).all()
