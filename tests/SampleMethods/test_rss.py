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
    strata = Rectangular(strata_number=[4, 4], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
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
    strata = Voronoi(seeds_number=16, dimension=2, random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
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
    strata = Rectangular(strata_number=[4, 4], random_state=1)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1)
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
