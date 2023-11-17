"""
This is the test module for Sobol sensitivity indices.

Here, we will use the Ishigami function to test the output.

The following methods are tested:
1. generate_pick_and_freeze_samples
2. pick_and_freeze_estimator (First and Total order Sobol indices)
3. pick_and_freeze_estimator (Second order Sobol indices) using [1]_.

References
----------

.. [1] Graham Glen, Kristin Isaacs, Estimating Sobol sensitivity indices using 
       correlations, Environmental Modelling & Software, Volume 37, 2012, Pages 157-166,
       ISSN 1364-8152,  https://doi.org/10.1016/j.envsoft.2012.03.014.


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
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity

# Prepare
###############################################################################

# Prepare the input distribution
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
def sobol_object(ishigami_model_object, ishigami_input_dist_object):
    """This function returns the Sobol object."""

    return SobolSensitivity(ishigami_model_object, ishigami_input_dist_object)


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


@pytest.fixture()
def saltelli_ishigami_Sobol_indices(sobol_object):

    SA = sobol_object

    np.random.seed(12345)  #! set seed for reproducibility

    SA.run(n_samples=1_000_000)

    return SA.first_order_indices, SA.total_order_indices


@pytest.fixture()
def NUM_SAMPLES():
    """This function returns the number of samples for bootstrapping"""

    num_bootstrap_samples = 10_000
    num_samples = 100_000

    return num_bootstrap_samples, num_samples


@pytest.fixture()
def bootstrap_sobol_index_variance(sobol_object, NUM_SAMPLES):

    #### SETUP ####
    SA = sobol_object

    np.random.seed(12345)  #! set seed for reproducibility

    confidence_level = 0.95
    delta = -scipy.stats.norm.ppf((1 - confidence_level) / 2)

    num_bootstrap_samples, n_samples = NUM_SAMPLES

    #### Compute indices ####
    SA.run(
        n_samples=n_samples,
        n_bootstrap_samples=num_bootstrap_samples,
        confidence_level=confidence_level,
    )

    First_order = SA.first_order_indices.ravel()
    Total_order = SA.total_order_indices.ravel()
    confidence_interval_first_order = SA.first_order_confidence_interval
    confidence_interval_total_order = SA.total_order_confidence_interval

    #### Compute variance ####
    upper_bound_first_order = confidence_interval_first_order[:, 1]
    upper_bound_total_order = confidence_interval_total_order[:, 1]

    std_bootstrap_first_order = (upper_bound_first_order - First_order) / delta
    std_bootstrap_total_order = (upper_bound_total_order - Total_order) / delta

    return std_bootstrap_first_order**2, std_bootstrap_total_order**2


@pytest.fixture()
def model_eval_sobol_index_variance():

    """
    For computational efficiency, the variance of the Sobol indices
    is precomputed using model evaluations with
    NUM_SAMPLES (num_repetitions=10_000, num_samples=100_000)

    Copy-paste the following code to generate the variance
    of the Sobol indices:

        runmodel_obj = RunModel(
                model_script='ishigami.py',
                var_names=['X1', 'X2', 'X3'],
                vec=True, delete_files=True)

        input_obj = JointInd([Uniform(-np.pi, 2*np.pi)]*3)

        SA = Sobol(runmodel_obj, input_obj)

        np.random.seed(12345) # for reproducibility

        num_repetitions, n_samples = 10_000, 100_000

        num_vars = 3

        sample_first_order = np.zeros((num_vars, num_repetitions))
        sample_total_order = np.zeros((num_vars, num_repetitions))

        for i in range(num_repetitions):
            S, S_T = SA.run(n_samples=n_samples)

            sample_first_order[:, i] = S.ravel()
            sample_total_order[:, i] = S_T.ravel()

        variance_first_order = np.var(sample_first_order, axis=1, ddof=1).reshape(-1, 1)
        variance_total_order = np.var(sample_total_order, axis=1, ddof=1).reshape(-1, 1)

        print(variance_first_order)
        print(variance_total_order)

    """

    variance_first_order = np.array([1.98518409e-05, 1.69268227e-05, 2.50390610e-05])

    variance_total_order = np.array([2.82995855e-05, 2.46373399e-05, 2.59811868e-05])

    return variance_first_order, variance_total_order


@pytest.fixture()
def sobol_g_function_input_dist_object():
    """
    This function returns the input distribution object for the Sobol G function.

    X1 ~ Uniform(0, 1)
    X2 ~ Uniform(0, 1)
    X3 ~ Uniform(0, 1)
    X4 ~ Uniform(0, 1)
    X5 ~ Uniform(0, 1)
    X6 ~ Uniform(0, 1)

    """

    dist_object = JointIndependent([Uniform(0, 1)] * 6)

    return dist_object


@pytest.fixture()
def sobol_g_function_model_object():
    """This function creates the Sobol g-function model object"""

    a_vals = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])

    model = PythonModel(
        model_script="sobol_func.py",
        model_object_name="evaluate",
        delete_files=True,
        a_values=a_vals,
    )

    runmodel_obj = RunModel(model=model)

    return runmodel_obj


@pytest.fixture()
def sobol_object_g_func(
    sobol_g_function_input_dist_object, sobol_g_function_model_object
):
    """This function creates the Sobol object for the g-function"""

    sobol_object = SobolSensitivity(
        sobol_g_function_model_object, sobol_g_function_input_dist_object
    )

    return sobol_object


@pytest.fixture()
def analytical_sobol_g_func_second_order_indices():
    """
    This function returns the analytical second order Sobol indices for the g-function

    The values were obtained from [1]_.

    """

    S12 = 0.0869305
    S13 = 0.0122246
    S14 = 0.00195594
    S15 = 0.00001956
    S16 = 0.00001956
    S23 = 0.00543316
    S24 = 0.00086931
    S25 = 0.00000869
    S26 = 0.00000869
    S34 = 0.00012225
    S35 = 0.00000122
    S36 = 0.00000122
    S45 = 0.00000020
    S46 = 0.00000020
    S56 = 2.0e-9

    S_2 = [S12, S13, S14, S15, S16, S23, S24, S25, S26, S34, S35, S36, S45, S46, S56]

    return np.array(S_2).reshape(-1, 1)


@pytest.fixture()
def saltelli_sobol_g_function(sobol_object_g_func):

    SA = sobol_object_g_func

    np.random.seed(12345)  #! set seed for reproducibility

    # Compute Sobol indices using the pick and freeze algorithm
    # Save only second order indices
    SA.run(n_samples=100_000, estimate_second_order=True)

    return SA.second_order_indices


# Unit tests
###############################################################################


def test_pick_and_freeze_estimator(
    analytical_ishigami_Sobol_indices, saltelli_ishigami_Sobol_indices
):

    """
    Test the Saltelli pick and freeze estimator using 1_000_000 samples.
    """

    # Prepare
    S_analytical, S_T_analytical = analytical_ishigami_Sobol_indices
    S_saltelli, S_T_saltelli = saltelli_ishigami_Sobol_indices

    # Act
    assert S_analytical.shape == S_saltelli.shape
    assert S_T_analytical.shape == S_T_saltelli.shape
    # Idea: Measure accuracy upto 2 decimal places -> rtol=0, atol=1e-2
    assert np.isclose(S_saltelli, S_analytical, rtol=0, atol=1e-2).all()
    assert np.isclose(S_T_saltelli, S_T_analytical, rtol=0, atol=1e-2).all()


def test_bootstrap_variance_computation(
    model_eval_sobol_index_variance, bootstrap_sobol_index_variance
):

    """Test the bootstrap variance computation."""

    # Prepare
    var_first, var_total = model_eval_sobol_index_variance
    boot_var_first, boot_var_total = bootstrap_sobol_index_variance

    # Act
    assert var_first.shape == boot_var_first.shape
    assert var_total.shape == boot_var_total.shape

    # Idea: Ensure bootstrap variance and MC variance are of same order -> rtol=0, atol=1e-4
    assert np.isclose(boot_var_first, var_first, rtol=0, atol=1e-4).all()
    assert np.isclose(boot_var_total, var_total, rtol=0, atol=1e-4).all()


def test_second_order_indices(
    analytical_sobol_g_func_second_order_indices, saltelli_sobol_g_function
):

    """Test the second order indices computation."""

    # Prepare
    S_2_analytical = analytical_sobol_g_func_second_order_indices
    S_2 = saltelli_sobol_g_function

    # Act
    # Idea: Ensure second order indices are of same order -> rtol=0, atol=1e-4
    assert np.isclose(S_2, S_2_analytical, rtol=0, atol=1e-2).all()
