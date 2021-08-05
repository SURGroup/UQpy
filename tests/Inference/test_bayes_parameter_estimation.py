import numpy as np
from UQpy.RunModel import RunModel # required to run the quadratic model
from sklearn.neighbors import KernelDensity # for the plots
from UQpy.distributions.collection import JointIndependent, Uniform, Lognormal, Normal
from UQpy.sampling import ImportanceSampling
from UQpy.inference.inference_models.DistributionModel import DistributionModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation


def pdf_from_kde(domain, samples1d):
    bandwidth = 1.06 * np.std(samples1d) * samples1d.size ** (-1/5)
    kde = KernelDensity(bandwidth=bandwidth).fit(samples1d.reshape((-1,1)))
    log_dens = kde.score_samples(domain)
    return np.exp(log_dens)


def test_probability_model():
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
    candidate_model = DistributionModel(distributions=Normal(loc=None, scale=None), nparams=2, prior=prior)

    sampling_class = ImportanceSampling()
    bayes_estimator = BayesParameterEstimation(inference_model=candidate_model, data=data,
                                               sampling_class=sampling_class,
                                               nsamples=10000, random_state=1)

    s_prior = bayes_estimator.sampler.samples
    bayes_estimator.sampler.resample()
    s_posterior = bayes_estimator.sampler.unweighted_samples

    assert s_posterior[0, 1] == 0.9977641464383822
    assert s_posterior[9999, 0] == 10.02449120238032