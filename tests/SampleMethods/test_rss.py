import pytest

from UQpy.distributions.collection.Uniform import Uniform
from UQpy.utilities.strata.Rectangular import Rectangular
from UQpy.sampling.RefinedStratifiedSampling import *
from UQpy.sampling.refined_stratified_sampling.SimpleRefinement import *
from UQpy.utilities.strata.Voronoi import *
from UQpy.RunModel import *
from UQpy.surrogates.kriging.Kriging import Kriging

import shutil


def test_rss_simple_rectangular():
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[4, 4])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    algorithm = SimpleRefinement(strata)
    y = RefinedStratifiedSampling(stratified_sampling=x,
                                  samples_number=18,
                                  samples_per_iteration=2,
                                  refinement_algorithm=algorithm,
                                  random_state=2)
    assert y.samples[16, 0] == 0.06614276178462988
    assert y.samples[16, 1] == 0.7836449863362334
    assert y.samples[17, 0] == 0.1891972651582183
    assert y.samples[17, 1] == 0.2961099664117288


def test_rss_simple_voronoi():
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Voronoi(seeds_number=16, dimension=2)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    algorithm = SimpleRefinement(strata)
    y = RefinedStratifiedSampling(stratified_sampling=x,
                                  samples_number=18,
                                  samples_per_iteration=2,
                                  refinement_algorithm=algorithm,
                                  random_state=2)
    assert y.samples[16, 0] == 0.44328265744393724
    assert y.samples[16, 1] == 0.4072924210691123
    assert y.samples[17, 0] == 0.3507629313878089
    assert y.samples[17, 1] == 0.17076741629044234


def test_gradient_enhanced_refinement_rectangular():
    from UQpy.surrogates.kriging.regression_models.Linear import Linear
    from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[4, 4])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    initial_samples = x.samples.copy()
    rmodel1 = RunModel(model_script='python_model_function.py', vec=False)
    rmodel1.run(samples=x.samples)
    num = 50
    x1 = np.linspace(0, 1, num)
    x2 = np.linspace(0, 1, num)
    x1v, x2v = np.meshgrid(x1, x2)
    y_act = np.zeros([num, num])
    r1model = RunModel(model_script='python_model_function.py')
    for i in range(num):
        for j in range(num):
            r1model.run(samples=np.array([[x1v[i, j], x2v[i, j]]]), append_samples=False)
            y_act[i, j] = r1model.qoi_list[0]

    regression_model = Linear()
    correlation_model = Exponential()
    K = Kriging(regression_model=regression_model, correlation_model=correlation_model,
                optimizations_number=20, correlation_model_parameters=[1, 1])
    K.fit(samples=x.samples, values=rmodel1.qoi_list)

    from UQpy.sampling.refined_stratified_sampling.GradientEnhancedRefinement import GradientEnhancedRefinement
    algorithm = GradientEnhancedRefinement(strata=strata,runmodel_object=rmodel1, surrogate=K)
    z = RefinedStratifiedSampling(stratified_sampling=x,
                                  refinement_algorithm=algorithm,
                                  random_state=2)
    z.run(samples_number=18)

    assert z.samples[16, 0] == 0.42949936276775047
    assert z.samples[16, 1] == 0.2564815579569728
    assert z.samples[17, 0] == 0.44370780973483864
    assert z.samples[17, 1] == 0.6088305981545692

    shutil.rmtree(r1model.model_dir)
    shutil.rmtree(rmodel1.model_dir)

# def test_rss3():
#     from UQpy.surrogates.kriging.regression_models.Linear import Linear
#     from UQpy.surrogates.kriging.correlation_models.Exponential import Exponential
#     marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
#     strata = Voronoi(seeds_number=16, dimension=2, random_state=1)
#     x = StratifiedSampling(distributions=marginals, strata_object=strata,
#                            samples_per_stratum_number=1)
#     rmodel1 = RunModel(model_script='python_model_function.py')
#     rmodel1.run(samples=x.samples)
#
#     num = 50
#     x1 = np.linspace(0, 1, num)
#     x2 = np.linspace(0, 1, num)
#     x1v, x2v = np.meshgrid(x1, x2)
#     y_act = np.zeros([num, num])
#     r1 = RunModel(model_script='python_model_function.py')
#     for i in range(num):
#         for j in range(num):
#             r1.run(samples=np.array([[x1v[i, j], x2v[i, j]]]))
#             y_act[i, j] = r1.qoi_list[-1]
#
#     k1 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
#     K = GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=5)
#
#     K.fit(x.samples, rmodel1.qoi_list)
#
#     from UQpy.sampling.refined_stratified_sampling.GradientEnhancedRefinement import GradientEnhancedRefinement
#     algorithm = GradientEnhancedRefinement(strata=strata, runmodel_object=rmodel1, surrogate=K)
#     z = RefinedStratifiedSampling(stratified_sampling=x,
#                                   refinement_algorithm=algorithm,
#                                   random_state=2)
#
#     z.run(nsamples=18)
#
#     assert z.samples[16, 0] == 0.42949936276775047
#     assert z.samples[16, 1] == 0.2564815579569728
#     assert z.samples[17, 0] == 0.44370780973483864
#     assert z.samples[17, 1] == 0.6088305981545692



marginals = [Uniform(loc=0., scale=2.), Uniform(loc=0., scale=1.)]
strata = Rectangular(strata_number=[2, 2])
x = StratifiedSampling(distributions=marginals, strata_object=strata, samples_per_stratum_number=1, random_state=1)
y = RefinedStratifiedSampling(sample_object=x, samples_number=6, n_add=2, random_state=2,
                              refinement_algorithm=SimpleRefinement())

# dir_path = os.path.dirname(os.path.realpath(__file__))
# filepath = os.path.join(dir_path, 'python_model_function.py')
rmodel = RunModel(model_script='python_model_function.py', vec='False')
from UQpy.surrogates.kriging.regression_models import Linear
from UQpy.surrogates.kriging.correlation_models import Exponential
K = Kriging(regression_model=Linear(), correlation_model=Exponential(), optimizations_number=20,
            correlation_model_parameters=[1, 1])
K.fit(samples=x.samples, values=rmodel.qoi_list)
# z = RefinedStratifiedSampling(sample_object=x, random_state=2,
#                               refinement_algorithm=GradientEnhancedRefinement()
#
#                               runmodel_object=rmodel, krig_object=K, random_state=2, max_train_size=4,
#                    verbose=True)
# z.run(nsamples=6)

strata_vor = Voronoi(seeds_number=4, dimension=2)
x_vor = StratifiedSampling(distributions=marginals, strata_object=strata_vor, samples_per_stratum_number=1,
                           random_state=10)
y_vor = RefinedStratifiedSampling(sample_object=x_vor, samples_number=6, n_add=2)

rmodel_ = RunModel(model_script='python_model_function.py', vec='False')
K_ = Kriging(reg_model='Linear', corr_model='Exponential', nopt=20, corr_model_params=[1, 1])
# K_.fit(samples=x_vor.samples, values=rmodel_.qoi_list)
# z_vor = VoronoiRSS(sample_object=x_vor, runmodel_object=rmodel_, krig_object=K_, nsamples=6,
#                    random_state=x_vor.random_state, max_train_size=4, verbose=True)


def test_rect_rss():
    """
    Test the 6 samples generated by RSS using rectangular stratification
    """
    tmp1 = (np.round(y.samples, 3) == np.array([[0.512, 0.475], [1.144, 0.474], [0.312, 0.712], [1.828, 0.705],
                                                [1.907, 0.046], [0.8, 0.864]])).all()
    tmp2 = (np.round(y.samplesU01, 3) == np.array([[0.256, 0.475], [0.572, 0.474], [0.156, 0.712], [0.914, 0.705],
                                                   [0.954, 0.046], [0.4, 0.864]])).all()
    assert tmp1 and tmp2


def test_rect_gerss():
    """
    Test the 6 samples generated by GE-RSS using rectangular stratification
    """
    tmp1 = (np.round(z.samples, 3) == np.array([[0.512, 0.475], [1.144, 0.474], [0.312, 0.712], [1.828, 0.705],
                                                [0.131, 0.149], [1.907, 0.046]])).all()
    tmp2 = (np.round(z.samplesU01, 3) == np.array([[0.256, 0.475], [0.572, 0.474], [0.156, 0.712], [0.914, 0.705],
                                                   [0.065, 0.149], [0.954, 0.046]])).all()
    assert tmp1 and tmp2


def test_vor_rss():
    """
    Test the 6 samples generated by RSS using voronoi stratification
    """
    tmp1 = (np.round(y_vor.samples, 3) == np.array([[1.563, 0.458], [1.741, 0.048], [0.757, 0.246], [0.396, 0.678],
                                                    [1.166, 0.762], [0.766, 0.508]])).all()
    tmp2 = (np.round(y_vor.samplesU01, 3) == np.array([[0.782, 0.458], [0.87, 0.048], [0.379, 0.246], [0.198, 0.678],
                                                       [0.583, 0.762], [0.383, 0.508]])).all()
    assert tmp1 and tmp2


def test_vor_gerss():
    """
    Test the 6 samples generated by GE-RSS using voronoi stratification
    """
    tmp1 = (np.round(z_vor.samples, 3) == np.array([[1.563, 0.458], [1.741, 0.048], [0.757, 0.246], [0.396, 0.678],
                                                    [1.077, 0.676], [0.503, 0.873]])).all()
    tmp2 = (np.round(z_vor.samplesU01, 3) == np.array([[0.782, 0.458], [0.87, 0.048], [0.379, 0.246], [0.198, 0.678],
                                                       [0.538, 0.676], [0.252, 0.873]])).all()
    assert tmp1 and tmp2


def test_rss_random_state():
    """
        Check 'random_state' is an integer or RandomState object.
    """
    with pytest.raises(TypeError):
        RefinedStratifiedSampling(sample_object=x, samples_number=6, n_add=2, random_state='abc')


def test_rss_runmodel_object():
    """
        Check 'runmodel_object' should be a UQpy.RunModel class object.
    """
    with pytest.raises(NotImplementedError):
        RefinedStratifiedSampling(sample_object=x, samples_number=6, n_add=2, runmodel_object='abc')


def test_rss_kriging_object():
    """
        Check 'kriging_object', it should have 'fit' and 'predict' methods.
    """
    with pytest.raises(NotImplementedError):
        RefinedStratifiedSampling(sample_object=x, samples_number=6, n_add=2, krig_object='abc', runmodel_object=rmodel_)


def test_nsamples():
    """
        Check 'nsamples' attributes, it should be an integer.
    """
    with pytest.raises(NotImplementedError):
        RefinedStratifiedSampling(sample_object=x, samples_number='a', n_add=2)