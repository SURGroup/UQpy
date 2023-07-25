import pytest

from UQpy import GaussianProcessRegression, LinearRegression
from UQpy.utilities.kernels.euclidean_kernels.RBF import RBF
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.collection import Normal
from UQpy.sampling.adaptive_kriging_functions import *


def test_akmcs_weighted_u():
    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)

    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=10, noise=False, regression_model=LinearRegression(),
                                    random_state=1)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -0.48297825309989356
    assert a.samples[20, 1] == 0.39006110248010434


def test_akmcs_u():

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=100, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -3.781937137406927
    assert a.samples[20, 1] == 0.17610325620498946


def test_akmcs_expected_feasibility():

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=100, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedFeasibility(eff_a=0, eff_epsilon=2, eff_stop=0.001)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 5.423754197908594
    assert a.samples[20, 1] == 2.0355505295053384


def test_akmcs_expected_improvement():

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=50, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovement()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[21, 0] == 6.878734574049913
    assert a.samples[20, 1] == -6.3410533857909215


def test_akmcs_expected_improvement_global_fit():

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=50, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovementGlobalFit()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -10.24267076486663
    assert a.samples[20, 1] == -11.419510366469687


def test_akmcs_samples_error():

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=50, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    with pytest.raises(NotImplementedError):
        a = AdaptiveKriging(distributions=[Normal(loc=0., scale=4.)] * 3, runmodel_object=rmodel, surrogate=gpr,
                            learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                            random_state=2, samples=x.samples)


def test_akmcs_u_run_from_init():
    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    kernel1 = RBF()
    bounds_1 = [[10 ** (-4), 10 ** 3], [10 ** (-3), 10 ** 2], [10 ** (-3), 10 ** 2]]
    optimizer1 = MinimizeOptimizer(method='L-BFGS-B', bounds=bounds_1)
    gpr = GaussianProcessRegression(kernel=kernel1, hyperparameters=[1, 10 ** (-3), 10 ** (-2)], optimizer=optimizer1,
                                    optimizations_number=100, noise=False, regression_model=LinearRegression(),
                                    random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=gpr,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2, nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -3.781937137406927
    assert a.samples[20, 1] == 0.17610325620498946
