from UQpy.Distributions import Normal
from UQpy.Reliability import FORM, SORM
import glob
import shutil
import numpy as np
import pytest


def model_i(samples):
    qoi_list = [0] * samples.shape[0]
    for i in range(samples.shape[0]):
        qoi_list[i] = samples[i, 0] * samples[i, 1] - 80
    return qoi_list


def test_sorm():
    dist1 = Normal(loc=500, scale=100)
    dist2 = Normal(loc=1000, scale=100)
    dist = [dist1, dist2]
    form_obj = FORM(dist_object=dist, runmodel_object=model_i)
    form_obj.run()
    sorm_obj = SORM(form_object=form_obj)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(sorm_obj.Pf_sorm, 2.8803e-7, rtol=1e-02)


def test_form_obj():
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert SORM(form_object='form')
