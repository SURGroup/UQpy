from UQpy.sensitivity.MorrisSensitivity import MorrisSensitivity
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import Uniform
import pytest
import shutil

#os.chdir('./tests/Sensitivity')


@pytest.fixture
def setup():
    a_values = [0.001, 99.]
    runmodel_object = RunModel(
        model_script='pfn.py', model_object_name='gfun_sensitivity', vec=True, a_values=a_values,
        var_names=['X{}'.format(i) for i in range(2)], delete_files=True)
    dist_object = [Uniform(), Uniform()]
    yield runmodel_object, dist_object
    shutil.rmtree(runmodel_object.model_dir)


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
