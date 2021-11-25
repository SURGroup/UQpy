import numpy as np
from sklearn.neighbors import KernelDensity  # for the plots

from UQpy.sampling.input_data.ISInput import *
from UQpy.distributions.collection import JointIndependent, Uniform, Lognormal, Normal
from UQpy.inference.inference_models.DistributionModel import DistributionModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.sampling.input_data.MhInput import MhInput


def pdf_from_kde(domain, samples1d):
    bandwidth = 1.06 * np.std(samples1d) * samples1d.size ** (-1 / 5)
    kde = KernelDensity(bandwidth=bandwidth).fit(samples1d.reshape((-1, 1)))
    log_dens = kde.score_samples(domain)
    return np.exp(log_dens)


def test_probability_model_importance_sampling():
    # Generate data from a probability model, here a Gaussian pdf, then learn its parameters,
    # mean and covariance, from this data

    np.random.seed(100)
    mu, sigma = 10, 1  # true mean and standard deviation
    np.random.seed(1)
    data = np.random.normal(mu, sigma, 100).reshape((-1, 1))

    p0 = Uniform(loc=0., scale=15)
    p1 = Lognormal(s=1., loc=0., scale=1.)
    prior = JointIndependent(marginals=[p0, p1])

    # create an instance of class Model
    candidate_model = DistributionModel(distributions=Normal(loc=None, scale=None),
                                        parameters_number=2, prior=prior)
    is_input = ISInput(random_state=1)
    bayes_estimator = BayesParameterEstimation \
        .create_with_importance_sampling(inference_model=candidate_model,
                                         is_input=is_input,
                                         data=data,
                                         samples_number=10000)
    bayes_estimator.sampler.resample()
    s_posterior = bayes_estimator.sampler.unweighted_samples

    assert s_posterior[0, 1] == 0.8616126410951304
    assert s_posterior[9999, 0] == 10.02449120238032


def test_probability_model_mcmc():
    np.random.seed(100)
    mu, sigma = 10, 1  # true mean and standard deviation
    np.random.seed(1)
    data = np.random.normal(mu, sigma, 100).reshape((-1, 1))

    p0 = Uniform(loc=0., scale=15)
    p1 = Lognormal(s=1., loc=0., scale=1.)
    prior = JointIndependent(marginals=[p0, p1])

    # create an instance of class Model
    candidate_model = DistributionModel(distributions=Normal(loc=None, scale=None),
                                        parameters_number=2, prior=prior)

    mh_input = MhInput(jump=10, burn_length=10, seed=[1.0, 0.2], random_state=1)

    bayes_estimator = BayesParameterEstimation \
        .create_with_mcmc_sampling(mcmc_input=mh_input,
                                   inference_model=candidate_model,
                                   data=data,
                                   samples_number=5)
    s = bayes_estimator.sampler.samples

    assert s[0, 1] == 3.5196936384257835
    assert s[1, 0] == 11.143811671048994
    assert s[2, 0] == 10.162512455643435
    assert s[3, 1] == 0.8541521389437781
    assert s[4, 1] == 1.0095454025762525


def test_example():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from UQpy.stochastic_process import KarhunenLoeveExpansion
    from UQpy.RunModel import RunModel

    I = np.load("data_beam.npz")
    xsen = I['arr_0']
    w = I['arr_1']
    youngs = I['arr_2']

    m = w.shape
    Xm, Ym = np.meshgrid(xsen, xsen)

    # Modeling of the noise
    mu_eps = np.zeros(m).reshape(1, -1)  # mean of the additive Gaussian noise
    sigma_eps = 1e-3;  # std of the measurement error [m]
    l_eps = 1;  # correlation length of the error [m]
    Sigma_ee = (sigma_eps ** 2) * np.exp(-abs(Xm - Ym) / l_eps);  # exponential kernel
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Xm, Ym, Sigma_ee, cmap=cm.coolwarm)
    ax.invert_xaxis()

    from UQpy.inference import ComputationalModel, BayesParameterEstimation
    from UQpy.distributions import Normal, JointIndependent, Uniform
    from UQpy.sampling.mcmc.ModifiedMetropolisHastings import ModifiedMetropolisHastings
    from UQpy.sampling.input_data.MmhInput import MmhInput

    dist_proposal = Normal()
    dist_prior = Uniform()

    prior = JointIndependent(marginals=[dist_prior] * 101)
    proposal = JointIndependent(marginals=[dist_proposal] * 101)

    run_model_object = RunModel(model_script='pfn_test.py', model_object_name='run_cantilever', delete_files=True)

    inference_model = ComputationalModel(parameters_number=101, runmodel_object=run_model_object, error_covariance=1.0,
                                         prior=prior)

    # Try it with error_covariance = Sigma_ee (2d array). It gives an error.

    mmh_input = MmhInput()
    mmh_input.jump = 1
    mmh_input.burn_length = 0
    mmh_input.chains_number = 3
    mmh_input.proposal = proposal
    mmh_input.random_state = 123

    # bayes_estimator = BayesParameterEstimation.create_with_mcmc_sampling(inference_model=inference_model,
    #                                                                      data=w.reshape(1, -1),
    #                                                                      mcmc_input=mmh_input,
    #                                                                      samples_number=100)
    bayes_estimator = BayesParameterEstimation.create_with_mcmc_sampling(inference_model=inference_model,
                                                                         data=w,
                                                                         mcmc_input=mmh_input,
                                                                         samples_number=100)

    s = bayes_estimator.sampler.samples
    mean_s = np.mean(s, axis=0)
    E = 200e9
    plt.plot(E * mean_s)
    plt.plot(E * youngs.flatten(), 'r')

def test_example1():
    from UQpy.inference import ComputationalModel, BayesParameterEstimation
    from UQpy.distributions import Normal, JointIndependent, Uniform
    from UQpy.sampling.mcmc.ModifiedMetropolisHastings import ModifiedMetropolisHastings
    from UQpy.sampling.input_data.MmhInput import MmhInput

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from UQpy.stochastic_process import KarhunenLoeveExpansion
    from UQpy.RunModel import RunModel

    # Generate data
    param_true = np.array([1.0, 2.0]).reshape((1, -1))
    print('Shape of true parameter vector: {}'.format(param_true.shape))

    h_func = RunModel(model_script='pfn_models.py', model_object_name='model_quadratic1', vec=False,
                      var_names=['theta_0', 'theta_1'])
    h_func.run(samples=param_true)
    data_clean = np.array(h_func.qoi_list[0])

    error_covariance = 1.
    noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50, random_state=123).reshape((50,))
    data_3 = data_clean + noise
    print('Shape of data: {}'.format(data_3.shape))
    print(data_3[:4])

    p0 = Normal()
    p1 = Normal()
    prior = JointIndependent(marginals=[p0, p1])

    inference_model = ComputationalModel(parameters_number=2, runmodel_object=h_func, error_covariance=error_covariance,
                                         prior=prior)

    proposal = JointIndependent([Normal(scale=0.1), Normal(scale=0.05)])

    mh_input1 = MhInput()
    mh_input1.jump = 10
    mh_input1.burn_length = 0
    mh_input1.proposal = proposal
    mh_input1.chains_number = 3
    mh_input1.random_state = 456
    bayes_estimator = BayesParameterEstimation.create_with_mcmc_sampling(inference_model=inference_model,
                                                                         data=data_3,
                                                                         mcmc_input=mh_input1,
                                                                         samples_number=500)