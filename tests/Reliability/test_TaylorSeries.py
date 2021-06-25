from UQpy.Distributions import Normal, JointInd
from UQpy.Reliability import TaylorSeries
from UQpy.Transformations import Nataf
import glob
import shutil
import numpy as np
import pytest

########################################################################################################################
# Tests for the constructor of the TaylorSeries class.


def model_i(samples):
    qoi_list = [0] * samples.shape[0]
    for i in range(samples.shape[0]):
        resistance = samples[i, 0]
        stress = samples[i, 1]
        qoi_list[i] = resistance - stress
    return qoi_list


def test_dist_object1():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist = [dist1]
    obj = TaylorSeries(dist_object=dist, runmodel_object=model_i)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    assert obj.dimension == 1


def test_dist_object2():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist = [dist1, 'Beta']
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert TaylorSeries(dist_object=dist, runmodel_object=model_i)


def test_dist_object3():
    dist1 = Normal(loc=0.0, scale=1.0)
    obj = TaylorSeries(dist_object=dist1, runmodel_object=model_i)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    assert obj.dimension == 1


def test_dist_object4():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    dist3 = Normal(loc=0.0, scale=1.0)
    obj = TaylorSeries(dist_object=[dist1, JointInd(marginals=[dist2, dist3])], runmodel_object=model_i)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    assert obj.dimension == 3


def test_dist_object5():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    dist3 = 'Beta'
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert TaylorSeries(dist_object=[dist1, JointInd(marginals=[dist2, dist3])], runmodel_object=model_i)

########################################################################################################################
# Tests for the derivatives function of the TaylorSeries class.


def test_derivatives_1_no_samples():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert TaylorSeries.derivatives(nataf_object=ntf_obj, runmodel_object=model_i)


def test_derivatives_3_no_nataf():
    point_u = np.array([-2, 1])
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert TaylorSeries.derivatives(point_u=point_u, runmodel_object=model_i)


def test_derivatives_4_callable_model():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    point_u = np.array([-2, 1])
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    gradient, qoi, array_of_samples = TaylorSeries.derivatives(point_u=point_u, runmodel_object=model_i,
                                                               nataf_object=ntf_obj)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(array_of_samples[0], [160, 160], rtol=1e-09)
    np.testing.assert_allclose(gradient, [20, -10], rtol=1e-09)


def test_derivatives_5_run_model():
    dist1 = Normal(loc=200, scale=20)
    dist2 = Normal(loc=150, scale=10)
    point_u = np.array([-2, 1])
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    gradient, qoi, array_of_samples = TaylorSeries.derivatives(point_u=point_u, runmodel_object=model_i,
                                                               nataf_object=ntf_obj)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(array_of_samples[0], [160, 160], rtol=1e-09)
    np.testing.assert_allclose(gradient, [20, -10], rtol=1e-09)


def model_j(samples):
    d0 = 3
    e = 30000000
    l = 100
    w = 2
    t = 4
    qoi_list = [0] * samples.shape[0]
    for i in range(samples.shape[0]):
        qoi_list[i] = d0 - 4*l**3/(e*w*t)*np.sqrt((samples[i, 1]/t**2)**2 + (samples[i, 0]/w**2)**2)
    return qoi_list


def test_derivatives_6_second():
    dist1 = Normal(loc=500, scale=100)
    dist2 = Normal(loc=1000, scale=100)
    point_u = np.array([1.73673009, 0.16383283])
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    hessian = TaylorSeries.derivatives(point_u=point_u, runmodel_object=model_j,
                                       nataf_object=ntf_obj, order='second')
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(hessian, [[-0.00720754, 0.00477726], [0.00477726, -0.00316643]], rtol=1e-04)


