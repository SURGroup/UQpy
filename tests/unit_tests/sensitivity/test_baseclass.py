"""
This module is used to test the functionalities of the baseclass.

- test_pick_and_freeze_sampling: 
    Test the `generate_pick_and_test_samples` function.
- test_bootstrap_for_vector: 
    Test the bootstrap sampling for a vector.
- test_bootstrap_for_matrix: 
    Test the bootstrap sampling for a matrix.

"""

import numpy as np
import pytest

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity
from UQpy.sensitivity.baseclass.PickFreeze import generate_pick_freeze_samples

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
def sobol_object_input_samples_small(sobol_object):
    """This creates the Sobol object."""

    SA = sobol_object

    np.random.seed(12345)  # set seed for reproducibility

    SA.n_samples = 2

    return generate_pick_freeze_samples(SA.dist_object, SA.n_samples)


# Generate N pick and free samples
@pytest.fixture()
def pick_and_freeze_samples_small():
    """
    This function returns input matrices A, B and C_i with a small number
    of samples for the Ishigami input distribution.
    This is used to test the `generate_pick_and_freeze_samples` function.

    The samples are generated as follows:

        dist_1 = JointInd([Uniform(-np.pi, 2*np.pi)]*3)

        np.random.seed(12345) #! set seed for reproducibility

        n_samples = 2
        n_vars = 3

        samples = dist_1.rvs(n_samples*2)

        # Split samples
        A_samples = samples[:n_samples, :]
        B_samples = samples[n_samples:, :]

        def _get_C_i(i, A, B):
            C_i = copy.deepcopy(B)
            C_i[:, i] = A[:, i]
            return C_i

        C_samples = np.zeros((n_vars, n_samples, n_vars))

        for i in range(3):
            C_samples[i, :, :] = _get_C_i(i, A_samples, B_samples)

        print(np.around(A_samples,3))
        print(np.around(B_samples,3))
        print(np.around(C_samples,3))

    """

    A_samples = np.array([[2.699, 0.426, 1.564], [-1.154, 0.600, 0.965]])

    B_samples = np.array([[-1.986, 2.919, 1.556], [-1.856, 0.962, 2.898]])

    C_samples = np.array(
        [
            [[2.699, 2.919, 1.556], [-1.154, 0.962, 2.898]],
            [[-1.986, 0.426, 1.556], [-1.856, 0.6, 2.898]],
            [[-1.986, 2.919, 1.564], [-1.856, 0.962, 0.965]],
        ]
    )

    return A_samples, B_samples, C_samples


@pytest.fixture()
def random_f_A():
    """This function returns an A-like vector"""

    rand_f_A = np.array([[100], [101], [102], [103], [104]])

    return rand_f_A


@pytest.fixture()
def random_f_C_i():
    """This function returns a C_i-like vector"""

    rand_f_C_i = np.array([[100, 200], [101, 201], [102, 202], [103, 203], [104, 204]])
    return rand_f_C_i


@pytest.fixture()
def manual_bootstrap_samples_f_A():
    """This function bootstraps the A-like vector using random indices"""

    # Genrated using np.random.randint(low=0, high=5, size=(5,1))
    # with np.random.seed(12345)
    # rand_indices_f_A = np.array([ [2],
    #                             [1],
    #                             [4],
    #                             [1],
    #                             [2]])

    # bootstrap_f_A = rand_f_A[rand_indices_A]
    bootstrap_sample_A = np.array([[102], [101], [104], [101], [102]])

    return bootstrap_sample_A


@pytest.fixture()
def manual_bootstrap_samples_f_C_i():
    """This function bootstraps the C_i-like vector using random indices"""

    # Genrated using np.random.randint(low=0, high=5, size=(5,2))
    # with np.random.seed(12345)
    # rand_indices_C_i = np.array([ [2, 1],
    #                               [4, 1],
    #                               [2, 1],
    #                               [1, 3],
    #                               [1, 3]])

    bootstrap_f_C_i = np.array(
        [[102, 201], [104, 201], [102, 201], [101, 203], [101, 203]]
    )

    return bootstrap_f_C_i


# Unit tests
###############################################################################


def test_pick_and_freeze_sampling(
    pick_and_freeze_samples_small, sobol_object_input_samples_small
):

    """Test the `generate_pick_and_test_samples` function."""

    # Prepare
    A_samples, B_samples, C_samples = pick_and_freeze_samples_small
    A_test, B_test, C_test_generator, _ = sobol_object_input_samples_small

    # Act
    assert np.allclose(A_samples, np.around(A_test, 3))
    assert np.allclose(B_samples, np.around(B_test, 3))

    for i in range(3):
        C_test = next(C_test_generator)
        assert np.allclose(C_samples[i, :, :], np.around(C_test, 3))


def test_bootstrap_for_vector(random_f_A, manual_bootstrap_samples_f_A):

    """Test the bootstrap sampling for a vector."""

    # Prepare
    np.random.seed(12345)  #! set seed for reproducibility

    gen_f_A = SobolSensitivity.bootstrap_sample_generator_1D(random_f_A)

    bootstrap_samples_f_A = next(gen_f_A)

    # Act
    assert np.array_equal(manual_bootstrap_samples_f_A, bootstrap_samples_f_A)


def test_bootstrap_for_matrix(random_f_C_i, manual_bootstrap_samples_f_C_i):

    """Test the bootstrap sampling for a matrix."""

    # Prepare
    np.random.seed(12345)  #! set seed for reproducibility

    gen_f_C_i = SobolSensitivity.bootstrap_sample_generator_2D(random_f_C_i)

    bootstrap_samples_C_i = next(gen_f_C_i)

    # Act
    assert np.array_equal(manual_bootstrap_samples_f_C_i, bootstrap_samples_C_i)
