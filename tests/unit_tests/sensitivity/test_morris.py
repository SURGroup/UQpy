from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.sensitivity.MorrisSensitivity import MorrisSensitivity
from UQpy.distributions import Uniform
import pytest


# os.chdir('./tests/Sensitivity')


@pytest.fixture
def setup():
    model = PythonModel(model_script='pfn.py', model_object_name='gfun_sensitivity', delete_files=True,
                        a_values=[0.001, 99.], var_names=['X{}'.format(i) for i in range(2)])
    runmodel_object = RunModel(model=model)
    dist_object = [Uniform(), Uniform()]
    yield runmodel_object, dist_object


def test_morris(setup):
    runmodel_object, dist_object = setup
    sens = MorrisSensitivity(runmodel_object=runmodel_object, distributions=dist_object, n_levels=8,
                             random_state=123, n_trajectories=3)
    assert round(sens.mustar_indices[1], 3) == 0.025


def test_morris_2(setup):
    runmodel_object, dist_object = setup
    sens = MorrisSensitivity(runmodel_object=runmodel_object, distributions=dist_object, n_levels=9,
                             random_state=123)
    sens.run(n_trajectories=2)
    sens.run(n_trajectories=2)
    assert round(sens.mustar_indices[1], 3) == 0.034


def test_morris_max_dispersion(setup):
    runmodel_object, dist_object = setup
    sens = MorrisSensitivity(runmodel_object=runmodel_object, distributions=dist_object, n_levels=9,
                             random_state=123, maximize_dispersion=True)
    sens.run(n_trajectories=5)
    assert round(sens.mustar_indices[1], 3) == 0.051
