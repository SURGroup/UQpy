"""
This is the test module for Cramer sensitivity indices.

Here, we will use the an exponential function to test the output.

The following methods are tested:
1. pick_and_freeze_estimator
2. bootstrap_variance_computation

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
from UQpy.distributions import Normal, Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.CramervonMises import CramervonMises

# Prepare
###############################################################################

# Prepare the input distribution
@pytest.fixture()
def exponential_input_dist_object():
    """
    This function returns the input distribution for the Ishigami function.

    X1 ~ Normal(0,1)
    X2 ~ Normal(0,1)

    """
    return JointIndependent([Normal(0, 1)] * 2)


@pytest.fixture()
def exponential_model_object():
    """This function creates the exponential run_model_object"""
    model = PythonModel(
        model_script="exponential.py",
        model_object_name="evaluate",
        var_names=[r"$X_1$", "$X_2$"],
        delete_files=True,
    )

    runmodel_obj = RunModel(model=model)

    return runmodel_obj


@pytest.fixture()
def CVM_object(exponential_model_object, exponential_input_dist_object):
    """This function returns the CVM object."""

    return CramervonMises(exponential_model_object, exponential_input_dist_object)


@pytest.fixture()
def analytical_exponential_CVM_indices():
    """This function returns the analytical Cramer-von-Mises indices.

    S1_CVM = (6/np.pi) * np.arctan(2) - 2
    S2_CVM = (6/np.pi) * np.arctan(np.sqrt(19)) - 2

    print(np.around(S1_CVM, 4))
    print(np.around(S2_CVM, 4))

    """

    return np.array([[0.1145], [0.5693]])


@pytest.fixture()
def numerical_exponential_CVM_indices(CVM_object):
    """
    This function returns the Cramer-von-Mises indices
    computed using the Pick and Freeze algorithm.

    """

    SA = CVM_object

    np.random.seed(12345)  #! set seed for reproducibility

    computed_indices = SA.run(n_samples=50_000)

    return computed_indices["CVM_i"]


@pytest.fixture()
def NUM_SAMPLES():
    """This function returns the number of samples."""

    num_bootstrap_samples = 50
    num_samples = 10_000

    return num_bootstrap_samples, num_samples


@pytest.fixture()
def bootstrap_CVM_index_variance(CVM_object, NUM_SAMPLES):
    """This function returns the variance in the computed Cramer-von-Mises index
    computed using the bootstrap algorithm."""

    #### SETUP ####
    SA = CVM_object

    np.random.seed(12345)  #! set seed for reproducibility

    confidence_level = 0.95
    delta = -scipy.stats.norm.ppf((1 - confidence_level) / 2)

    num_bootstrap_samples, n_samples = NUM_SAMPLES

    #### Compute indices ####
    computed_indices = SA.run(
        n_samples=n_samples,
        num_bootstrap_samples=num_bootstrap_samples,
        confidence_level=confidence_level,
    )

    First_order = computed_indices["CVM_i"].ravel()
    upper_bound_first_order = computed_indices["CI_CVM_i"][:, 1]

    #### Compute variance ####
    std_bootstrap_first_order = (upper_bound_first_order - First_order) / delta

    return std_bootstrap_first_order**2


@pytest.fixture()
def model_evals_CVM_index_variance():

    """
    runmodel_obj = RunModel(
                model_script='exponential.py',
                var_names=['X1', 'X2'],
                vec=True, delete_files=True)

    input_object = JointInd([Normal(0, 1)]*2)

    SA = CramervonMises(runmodel_obj, input_object)

    np.random.seed(12345)

    num_repetitions, n_samples = 1_000, 10_000

    num_vars = 2

    sample_first_order = np.zeros((num_vars, num_repetitions))

    for i in range(num_repetitions):
        CV_First_order = SA.run(n_samples=n_samples)

        sample_first_order[:, i] = CV_First_order.ravel()

    variance_first_order = np.var(sample_first_order, axis=1).reshape(-1, 1)

    print(variance_first_order)

    """

    variance_first_order = np.array([4.01099066e-05, 2.06802165e-05])

    return variance_first_order


@pytest.fixture()
def ishigami_input_dist_object():
    """
    This function returns the input distribution for the Ishigami function.

    X1 ~ Uniform(-pi, pi)
    X2 ~ Uniform(-pi, pi)
    X3 ~ Uniform(-pi, pi)

    """
    return JointIndependent([Uniform(-np.pi, 2 * np.pi)] * 3)


@pytest.fixture()
def ishigami_model_object():
    """This function creates the Ishigami run_model_object"""
    model = PythonModel(
        model_script="ishigami.py",
        model_object_name="evaluate",
        var_names=[r"$X_1$", "$X_2$", "$X_3$"],
        delete_files=True,
        params=[7, 0.1],
    )

    runmodel_obj = RunModel(model=model)

    return runmodel_obj


@pytest.fixture()
def CVM_object_ishigami(ishigami_model_object, ishigami_input_dist_object):
    """This function returns the CVM object."""

    return CramervonMises(ishigami_model_object, ishigami_input_dist_object)


@pytest.fixture()
def numerical_Sobol_indices(CVM_object_ishigami):
    """
    This function returns the Sobol indices computed
    using the Pick and Freeze algorithm.
    """

    SA = CVM_object_ishigami

    np.random.seed(12345)

    computed_indices = SA.run(
        n_samples=500_000, estimate_sobol_indices=True, disable_CVM_indices=True
    )

    return computed_indices["sobol_i"], computed_indices["sobol_total_i"]


@pytest.fixture()
def analytical_ishigami_Sobol_indices():
    """
    Analytical Sobol indices for the Ishigami function.

    Copy-paste the following to reproduce the given indices:

        a = 7
        b = 0.1

        V1 = 0.5*(1 + (b*np.pi**4)/5)**2
        V2 = (a**2)/8
        V3 = 0

        VT3 = (8*(b**2)*np.pi**8)/225
        VT1 = V1 + VT3
        VT2 = V2

        total_variance = V2 + (b*np.pi**4)/5 + ((b**2) * np.pi**8)/18 + 0.5

        S = np.array([V1, V2, V3])/total_variance
        S_T = np.array([VT1, VT2, VT3])/total_variance

        S = np.around(S, 4)
        S_T = np.around(S_T, 4)

    """

    S1 = 0.3139
    S2 = 0.4424
    S3 = 0

    S_T1 = 0.5576
    S_T2 = 0.4424
    S_T3 = 0.2437

    S = np.array([S1, S2, S3])
    S_T = np.array([S_T1, S_T2, S_T3])

    return S.reshape(-1, 1), S_T.reshape(-1, 1)


# Unit tests
###############################################################################


def test_pick_and_freeze_estimator(
    numerical_exponential_CVM_indices, analytical_exponential_CVM_indices
):
    """
    This function tests the pick_and_freeze_estimator method using 50_000 samples.
    """
    S_CVM_analytical = analytical_exponential_CVM_indices
    S_CVM_numerical = numerical_exponential_CVM_indices

    assert np.isclose(S_CVM_analytical, S_CVM_numerical, rtol=0, atol=1e-2).all()


def test_bootstrap_variance_computation(
    bootstrap_CVM_index_variance, model_evals_CVM_index_variance
):
    """
    This function tests the bootstrap_variance_computation method using
    100_000 samples and 1_000 bootstrap samples.
    """
    var_first = model_evals_CVM_index_variance
    boot_var_first = bootstrap_CVM_index_variance

    assert var_first.shape == boot_var_first.shape
    assert np.isclose(boot_var_first, var_first, rtol=0, atol=1e-4).all()


def test_Sobol_estimate_computation(
    numerical_Sobol_indices, analytical_ishigami_Sobol_indices
):
    """
    This function tests the Sobol_estimate_computation method using 1_000_000 samples.
    """
    S_numerical, S_T_numerical = numerical_Sobol_indices
    S_analytical, S_T_analytical = analytical_ishigami_Sobol_indices

    assert S_analytical.shape == S_numerical.shape
    assert S_T_analytical.shape == S_T_numerical.shape
    assert np.isclose(S_numerical, S_analytical, rtol=0, atol=1e-2).all()
    assert np.isclose(S_T_numerical, S_T_analytical, rtol=0, atol=1e-2).all()
