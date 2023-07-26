from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import *
from UQpy.reliability import FORM
import glob
import shutil
import numpy as np
import pytest
import os


@pytest.fixture
def setup():
    model = PythonModel(model_script='pfn1.py', model_object_name='model_i', delete_files=True)
    h_func = RunModel(model=model)
    yield h_func
    # shutil.rmtree(h_func.model_dir)


def test_seeds_xu_is_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.u_record[0][0][0], [0., 0.], rtol=1e-02)


def test_seeds_x_is_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1])
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_seed_u_is_none(setup):
    """ToDo: Fix FORM.run(seed_x=numpy_array) to pass this test"""
    distributions = [Normal(loc=200, scale=20), Normal(loc=150, scale=10)]
    form = FORM(distributions=distributions, runmodel_object=setup)
    seed_x = np.array([225, 140])
    form.run(seed_x=seed_x)
    np.testing.assert_allclose(form.failure_probability, 0.0126, rtol=1e-02)


def test_tol1_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_u=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol2_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_beta=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol3_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_gradient=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol12_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_u=1.0e-3, tolerance_beta=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol13_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_u=1.0e-3, tolerance_gradient=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol23_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_gradient=1.0e-3, tolerance_beta=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_tol123_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tolerance_u=1.0e-3, tolerance_gradient=1.0e-3, tolerance_beta=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_form_example():
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    model = PythonModel(model_script='pfn3.py', model_object_name='example1', delete_files=True)
    RunModelObject = RunModel(model=model)
    dist1 = Normal(loc=200., scale=20.)
    dist2 = Normal(loc=150, scale=10.)
    Q = FORM(distributions=[dist1, dist2], runmodel_object=RunModelObject,
             tolerance_u=1e-5, tolerance_beta=1e-5)
    Q.run()

    # print results
    np.allclose(Q.design_point_u, np.array([-2., 1.]))
    np.allclose(Q.design_point_x, np.array([160., 160.]))
    assert Q.beta[0] == 2.236067977499917
    assert Q.failure_probability[0] == 0.012673659338729965
    np.allclose(Q.state_function_gradient_record, np.array([0., 0.]))

