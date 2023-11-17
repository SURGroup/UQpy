from UQpy.distributions import Normal
from UQpy.reliability import FORM, SORM
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
import glob
import shutil
import numpy as np
import pytest
import os


@pytest.fixture
def setup():
    path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    model = PythonModel(model_script='pfn4.py', model_object_name='model_k', delete_files=True)
    h_func = RunModel(model=model)
    yield h_func


def test_sorm(setup):
    dist1 = Normal(loc=500, scale=100)
    dist2 = Normal(loc=1000, scale=100)
    dist = [dist1, dist2]
    form_obj = FORM(distributions=dist, runmodel_object=setup)
    form_obj.run()
    sorm_obj = SORM(form_object=form_obj)
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    np.testing.assert_allclose(sorm_obj.failure_probability, 2.8803e-7, rtol=1e-02)

def test_form_obj():
    for file_name in glob.glob("Model_Runs_*"):
        shutil.rmtree(file_name)
    with pytest.raises(Exception):
        assert SORM(form_object='form')




