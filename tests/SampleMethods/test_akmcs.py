from UQpy.surrogates.kriging.Kriging import Kriging
from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.RunModel import RunModel
from UQpy.distributions.collection import Normal
from UQpy.sampling.adaptive_kriging_functions import *


def test_akmcs_weighted_u():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 3.3036943922280737
    assert a.samples[20, 1] == -0.16784257369267955


def test_akmcs_u():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 2.573098622361529
    assert a.samples[20, 1] == -7.865501626326106


def test_akmcs_expected_feasibility():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedFeasibility(eff_a=0, eff_epsilon=2, eff_stop=0.001)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 1.366058523912817
    assert a.samples[20, 1] == -12.914668932772358


def test_akmcs_expected_improvement():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovement()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 4.553078100499578
    assert a.samples[20, 1] == -3.508949564718469


def test_akmcs_Expected_improvement_global_fit():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovementGlobalFit()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 11.939859785098493
    assert a.samples[20, 1] == -8.429899469300118