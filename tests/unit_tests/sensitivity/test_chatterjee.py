"""
This is the test module for the Chatterjee sensitivity indices. 

Here, we will use the exponential function to test the output, as in
the test module for Cramer sensitivity indices for the Chatterjee indices and 
the ishigami function as in the test module for Sobol sensitivity indices for the
Sobol indices.

The following methods are tested:
1. pick_and_freeze_estimator
2. Sobol estimate

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

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform, Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.ChatterjeeSensitivity import ChatterjeeSensitivity


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
        var_names=[
            "X_1",
            "X_2",
        ],
        delete_files=True,
    )

    runmodel_obj = RunModel(model=model)

    return runmodel_obj


@pytest.fixture()
def Chatterjee_object(exponential_model_object, exponential_input_dist_object):
    """This function creates the Chatterjee object"""
    return ChatterjeeSensitivity(exponential_model_object, exponential_input_dist_object)


@pytest.fixture()
def analytical_Chatterjee_indices():
    """This function returns the analytical Chatterjee indices.

    S1 = (6/np.pi) * np.arctan(2) - 2
    S2 = (6/np.pi) * np.arctan(np.sqrt(19)) - 2

    print(np.around(S1, 4))
    print(np.around(S2, 4))

    """

    return np.array([[0.1145], [0.5693]])


@pytest.fixture()
def numerical_Chatterjee_indices(Chatterjee_object):
    """This function returns the numerical Chatterjee indices."""

    SA = Chatterjee_object

    np.random.seed(12345)  #! set seed for reproducibility

    SA.run(n_samples=10_000)

    return SA.first_order_chatterjee_indices


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
def Chatterjee_object_ishigami(ishigami_model_object, ishigami_input_dist_object):
    """This function creates the Chatterjee object"""
    return ChatterjeeSensitivity(ishigami_model_object, ishigami_input_dist_object)


@pytest.fixture()
def numerical_Sobol_indices(Chatterjee_object_ishigami):
    """This function returns the Sobol indices."""

    SA = Chatterjee_object_ishigami

    np.random.seed(12345)

    SA.run(n_samples=10_000, estimate_sobol_indices=True)

    return SA.first_order_sobol_indices


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

    return S.reshape(-1, 1)


# Unit tests
###############################################################################


def test_Chatterjee_estimate(
    numerical_Chatterjee_indices, analytical_Chatterjee_indices
):
    """This function tests the Chatterjee estimate."""
    assert np.isclose(
        numerical_Chatterjee_indices, analytical_Chatterjee_indices, rtol=0, atol=1e-2
    ).all()


def test_Sobol_estimate(numerical_Sobol_indices, analytical_ishigami_Sobol_indices):
    """This function tests the Sobol estimate."""
    assert np.isclose(
        numerical_Sobol_indices, analytical_ishigami_Sobol_indices, rtol=0, atol=1e-2
    ).all()
