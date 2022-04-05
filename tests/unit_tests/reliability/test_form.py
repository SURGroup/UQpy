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
    h_func = RunModel(model_script='pfn.py', model_object_name='model_i', vec=False, delete_files=True)
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


def test_tol1_is_not_none(setup):
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol1=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol2=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol3=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol1=1.0e-3, tol2=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol1=1.0e-3, tol3=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol3=1.0e-3, tol2=1.0e-3)
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
    form_obj = FORM(distributions=dist, runmodel_object=setup, seed_u=[1, 1], tol1=1.0e-3, tol3=1.0e-3, tol2=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.failure_probability, 0.0126, rtol=1e-02)


def test_form_example():
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    RunModelObject = RunModel(model_script='pfn.py',
                              model_object_name="example1",
                              vec=False, ntasks=3)
    dist1 = Normal(loc=200., scale=20.)
    dist2 = Normal(loc=150, scale=10.)
    Q = FORM(distributions=[dist1, dist2], runmodel_object=RunModelObject,
             tol1=1e-5, tol2=1e-5)
    Q.run()

    # print results
    np.allclose(Q.DesignPoint_U, np.array([-2., 1.]))
    np.allclose(Q.DesignPoint_X, np.array([160., 160.]))
    assert Q.beta[0] == 2.236067977499917
    assert Q.failure_probability[0] == 0.012673659338729965
    np.allclose(Q.dg_u_record, np.array([0., 0.]))

    import shutil
    shutil.rmtree(RunModelObject.model_dir)
