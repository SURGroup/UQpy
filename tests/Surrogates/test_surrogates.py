from UQpy.surrogates.kriging.Kriging import Kriging
from UQpy.utilities.strata.Rectangular import Rectangular
from UQpy.sampling.StratifiedSampling import StratifiedSampling
from UQpy.RunModel import RunModel
from UQpy.distributions.collection.Gamma import Gamma
from UQpy.distributions.collection.Uniform import Uniform
import numpy as np
import shutil

def test_kriging_constant_exponential():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 5) == 3.99253
    assert round(K.correlation_model_parameters[1], 5) == 0.78878

    shutil.rmtree(rmodel.model_dir)


def test_kriging_linear_gaussian():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Gaussian import Gaussian
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Linear()
    correlation_model = Gaussian()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 52.625
    assert round(K.correlation_model_parameters[1], 3) == 3.027
    shutil.rmtree(rmodel.model_dir)


def test_kriging_quadratic_linear():
    from UQpy.surrogates.kriging.regression_models.Quadratic import Quadratic
    from UQpy.surrogates.kriging.correlation_models.Linear import Linear
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Quadratic()
    correlation_model = Linear()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 12.44
    assert round(K.correlation_model_parameters[1], 3) == 0.6
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_spherical():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Spherical import Spherical
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Spherical()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 2.315
    assert round(K.correlation_model_parameters[1], 3) == 0.536
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_spline():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Spline import Spline
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Spline()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 2.104
    assert round(K.correlation_model_parameters[1], 3) == 0.387
    shutil.rmtree(rmodel.model_dir)


def test_kriging_constant_cubic():
    from UQpy.surrogates.kriging.regression_models.Constant import Constant
    from UQpy.surrogates.kriging.correlation_models.Cubic import Cubic
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[10, 10], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
    rmodel = RunModel(model_script='python_model_function.py', vec=False)
    rmodel.run(samples=x.samples)

    regression_model = Constant()
    correlation_model = Cubic()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel.qoi_list)
    assert round(K.correlation_model_parameters[0], 3) == 17.419
    assert round(K.correlation_model_parameters[1], 3) == 0.275
    shutil.rmtree(rmodel.model_dir)
