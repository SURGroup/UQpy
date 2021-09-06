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
    is_input = ISInput()
    is_input.random_state=1
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

    mh_input = MhInput()
    mh_input.jump = 10
    mh_input.burn_length = 10
    mh_input.seed = np.array([1.0, 0.2])

    bayes_estimator = BayesParameterEstimation \
        .create_with_mcmc_sampling(mcmc_input=mh_input,
                                   inference_model=candidate_model,
                                   data=data,
                                   samples_number=5)
    s = bayes_estimator.sampler.samples

    assert s[0, 1] == 7.068887940481888
    assert s[1, 0] == 6.372598665842243
    assert s[2, 0] == 8.254666345432273
    assert s[3, 1] == 2.056254330693403
    assert s[4, 1] == 1.1035303119474327
