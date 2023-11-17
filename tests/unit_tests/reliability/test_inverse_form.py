import os
import pytest
import numpy as np
from scipy import stats
from UQpy.distributions import Normal
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.reliability.taylor_series import InverseFORM


@pytest.fixture
def inverse_form():
    """Example 7.2 from Chapter 7 of X. Du 2005

    Distributions are :math:`P_X \\sim N(500, 100)`, :math:`P_Y \\sim N(1000, 100)`
    Solution from the reference is :math:`u^*=(1.7367, 0.16376)`.
    Tolerances of :math:`1e-5` are used to ensure convergence to level of precision given by Du.
    """
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    python_model = PythonModel(model_script='example_7_2.py',
                               model_object_name='performance_function',
                               delete_files=True)
    runmodel_object = RunModel(model=python_model)
    distributions = [Normal(loc=500, scale=100), Normal(loc=1_000, scale=100)]
    return InverseFORM(distributions=distributions,
                       runmodel_object=runmodel_object,
                       p_fail=0.04054,
                       tolerance_u=1e-5,
                       tolerance_gradient=1e-5)


def test_no_seed(inverse_form):
    inverse_form.run()
    assert np.allclose(inverse_form.design_point_u, np.array([1.7367, 0.16376]), atol=1e-4)


def test_seed_x(inverse_form):
    seed_x = np.array([625, 900])
    inverse_form.run(seed_x=seed_x)
    assert np.allclose(inverse_form.design_point_u, np.array([1.7367, 0.16376]), atol=1e-4)


def test_seed_u(inverse_form):
    seed_u = np.array([2.4, -1.0])
    inverse_form.run(seed_u=seed_u)
    assert np.allclose(inverse_form.design_point_u, np.array([1.7367, 0.16376]), atol=1e-4)


def test_both_seeds(inverse_form):
    """Expected behavior is to raise ValueError and inform user only one input may be provided"""
    seed_x = np.array([1, 2])
    seed_u = np.array([3, 4])
    with pytest.raises(ValueError, match='UQpy: Only one input .* may be provided'):
        inverse_form.run(seed_u=seed_u, seed_x=seed_x)


def test_neither_tolerance():
    """Expected behavior is to raise ValueError and inform user at least one tolerance must be provided"""
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    python_model = PythonModel(model_script='example_7_2.py',
                               model_object_name='performance_function',
                               delete_files=True)
    runmodel_object = RunModel(model=python_model)
    distributions = [Normal(loc=500, scale=100), Normal(loc=1_000, scale=100)]
    with pytest.raises(ValueError, match='UQpy: At least one tolerance .* must be provided'):
        inverse_form = InverseFORM(distributions=distributions,
                                   runmodel_object=runmodel_object,
                                   p_fail=0.04054,
                                   tolerance_u=None,
                                   tolerance_gradient=None)


def test_beta():
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    python_model = PythonModel(model_script='example_7_2.py',
                               model_object_name='performance_function',
                               delete_files=True)
    runmodel_object = RunModel(model=python_model)
    distributions = [Normal(loc=500, scale=100), Normal(loc=1_000, scale=100)]
    inverse_form = InverseFORM(distributions=distributions,
                               runmodel_object=runmodel_object,
                               p_fail=None,
                               beta=-stats.norm.ppf(0.04054))
    inverse_form.run()
    assert np.allclose(inverse_form.design_point_u, np.array([1.7367, 0.16376]), atol=1e-3)


def test_no_beta_no_pfail():
    """Expected behavior is to raise ValueError and inform the user exactly one in put must be provided"""
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    python_model = PythonModel(model_script='example_7_2.py',
                               model_object_name='performance_function',
                               delete_files=True)
    runmodel_object = RunModel(model=python_model)
    distributions = [Normal(loc=500, scale=100), Normal(loc=1_000, scale=100)]
    with pytest.raises(ValueError, match='UQpy: Exactly one input .* must be provided'):
        inverse_form = InverseFORM(distributions=distributions,
                                   runmodel_object=runmodel_object,
                                   p_fail=None,
                                   beta=None)

