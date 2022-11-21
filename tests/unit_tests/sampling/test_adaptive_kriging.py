import pytest

from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.collection import Normal
from UQpy.sampling.adaptive_kriging_functions import *
import shutil


def test_akmcs_weighted_u():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizer=MinimizeOptimizer('l-bfgs-b'),
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=1)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 1.083176685073489
    assert a.samples[20, 1] == 0.20293978126855253



def test_akmcs_u():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizer=MinimizeOptimizer('l-bfgs-b'),
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -4.141979058326188
    assert a.samples[20, 1] == -1.6476534435429009



def test_akmcs_expected_feasibility():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1],
                optimizer=MinimizeOptimizer('l-bfgs-b'),)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedFeasibility(eff_a=0, eff_epsilon=2, eff_stop=0.001)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 1.366058523912817
    assert a.samples[20, 1] == -12.914668932772358



def test_akmcs_expected_improvement():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1],
                optimizer=MinimizeOptimizer('l-bfgs-b'),)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovement()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 4.553078100499578
    assert a.samples[20, 1] == -3.508949564718469



def test_akmcs_expected_improvement_global_fit():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1],
                optimizer=MinimizeOptimizer('l-bfgs-b'),)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovementGlobalFit()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 11.939859785098493
    assert a.samples[20, 1] == -8.429899469300118


def test_akmcs_samples_error():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizer=MinimizeOptimizer('l-bfgs-b'),
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=1)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    with pytest.raises(NotImplementedError):
        a = AdaptiveKriging(distributions=[Normal(loc=0., scale=4.)]*3, runmodel_object=rmodel, surrogate=K,
                            learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                            random_state=2, samples=x.samples)


def test_akmcs_u_run_from_init():
    from UQpy.surrogates.kriging.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.kriging.correlation_models.ExponentialCorrelation import ExponentialCorrelation

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = ExponentialCorrelation()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizer=MinimizeOptimizer('l-bfgs-b'),
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10**3, n_add=1, learning_function=learning_function,
                        random_state=2, nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -4.141979058326188
    assert a.samples[20, 1] == -1.6476534435429009
