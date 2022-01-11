import numpy as np
from sklearn.neighbors import KernelDensity  # for the plots

from UQpy.distributions.collection import JointIndependent, Uniform, Lognormal, Normal
from UQpy.inference.inference_models.DistributionModel import DistributionModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.sampling.ImportanceSampling import ImportanceSampling
from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings


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

    sampling = ImportanceSampling.create_for_inference(candidate_model, data, random_state=1)

    bayes_estimator = BayesParameterEstimation(sampling_class=sampling,
                                               inference_model=candidate_model,
                                               data=data,
                                               nsamples=10000)
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

    sampling = MetropolisHastings.create_for_inference(inference_model=candidate_model,
                                                       data=data,
                                                       jump=10, burn_length=10, seed=[1.0, 0.2], random_state=1)
    bayes_estimator = BayesParameterEstimation(sampling_class=sampling,
                                               inference_model=candidate_model,
                                               data=data,
                                               nsamples=5)
    s = bayes_estimator.sampler.samples

    assert s[0, 1] == 3.5196936384257835
    assert s[1, 0] == 11.143811671048994
    assert s[2, 0] == 10.162512455643435
    assert s[3, 1] == 0.8541521389437781
    assert s[4, 1] == 1.0095454025762525
