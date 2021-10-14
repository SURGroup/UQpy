

from matplotlib import pyplot as plt

from UQpy.surrogates.kriging.Kriging import Kriging
from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.RunModel import RunModel
from UQpy.distributions.collection import Normal
from UQpy.sampling.adaptive_kriging_functions import *
import shutil


def test_akmcs_weighted_u():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=0)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=1)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == 1.083176685073489
    assert a.samples[20, 1] == 0.20293978126855253

    shutil.rmtree(rmodel.model_dir)


def test_akmcs_u():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, samples_number=20, random_state=1)
    rmodel = RunModel(model_script='series.py', vec=False)
    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=10, correlation_model_parameters=[1, 1], random_state=0)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_samples_number=10**3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(samples_number=25, samples=x.samples)

    assert a.samples[23, 0] == -4.141979058326188
    assert a.samples[20, 1] == -1.6476534435429009

    shutil.rmtree(rmodel.model_dir)


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

    shutil.rmtree(rmodel.model_dir)


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

    shutil.rmtree(rmodel.model_dir)


def test_akmcs_expected_improvement_global_fit():
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

    shutil.rmtree(rmodel.model_dir)

def test_something():
    from UQpy.stochastic_process import BispectralRepresentation
    import numpy as np
    from scipy.stats import skew
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    n_sim = 100  # Num of samples

    n = 1  # Num of dimensions

    # Input parameters
    T = 600  # Time(1 / T = dw)
    nt = 12000  # Num.of Discretized Time
    F = 1 / T * nt / 2  # Frequency.(Hz)
    nf = 6000  # Num of Discretized Freq.

    # # Generation of Input Data(Stationary)
    dt = T / nt
    t = np.linspace(0, T - dt, nt)
    df = F / nf
    f = np.linspace(0, F - df, nf)

    S = 32 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)

    # Generating the 2 dimensional mesh grid
    fx = f
    fy = f
    Fx, Fy = np.meshgrid(f, f)

    b = 95 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
    B_Real = b
    B_Imag = b

    B_Real[0, :] = 0
    B_Real[:, 0] = 0
    B_Imag[0, :] = 0
    B_Imag[:, 0] = 0

    B_Complex = B_Real + 1j * B_Imag
    B_Ampl = np.absolute(B_Complex)

    t_u = 2 * np.pi / 2 / F

    if dt > t_u:
        print('Error')

    BSRM_object = BispectralRepresentation(n_sim, S, B_Complex, [dt], [df], [nt], [nf])
    samples = BSRM_object.samples

    fig, ax = plt.subplots()
    plt.title('Realisation of the BiSpectral Representation Method')
    plt.plot(t, samples[0, 0])
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.show()

    print('The mean of the samples is ', np.mean(samples), 'whereas the expected mean is 0.000')
    print('The variance of the samples is ', np.var(samples), 'whereas the expected variance is ', np.sum(S) * df * 2)
    print('The skewness of the samples is ', np.mean(skew(samples, axis=0)), 'whereas the expected skewness is ',
          np.sum(B_Real) * df ** 2 * 6 / (np.sum(S) * df * 2) ** (3 / 2))








































































































































































