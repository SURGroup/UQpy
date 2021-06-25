from UQpy.Distributions import Normal, JointInd
from UQpy.Reliability import FORM
import glob
import shutil
import numpy as np
import pytest


def model_i(samples):
    qoi_list = [0] * samples.shape[0]
    for i in range(samples.shape[0]):
        resistance = samples[i, 0]
        stress = samples[i, 1]
        qoi_list[i] = resistance - stress
    return qoi_list


def test_seeds_xu_is_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.u_record[0][0][0], [0., 0.], rtol=1e-02)


def test_seeds_x_is_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1])
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol1_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol1=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol2_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol2=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol3_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol3=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol12_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol1=1.0e-3, tol2=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol13_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol1=1.0e-3, tol3=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol23_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol3=1.0e-3, tol2=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)


def test_tol123_is_not_none():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i, seed_u=[1, 1], tol1=1.0e-3, tol3=1.0e-3, tol2=1.0e-3)
    form_obj.run()
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(form_obj.Pf_form, 0.0126, rtol=1e-02)
