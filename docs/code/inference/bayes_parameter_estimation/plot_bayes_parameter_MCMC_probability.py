"""

Parameter estimation using MCMC - Probability Model
=============================================================================

In the following we learn the mean and covariance of a univariate gaussian distribution from data.

"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

import numpy as np
import matplotlib.pyplot as plt

from UQpy.inference import DistributionModel, BayesParameterEstimation
from sklearn.neighbors import KernelDensity  # for the plots
from UQpy.distributions import JointIndependent, Uniform, Lognormal, Normal

def pdf_from_kde(domain, samples1d):
    bandwidth = 1.06 * np.std(samples1d) * samples1d.size ** (-1 / 5)
    kde = KernelDensity(bandwidth=bandwidth).fit(samples1d.reshape((-1, 1)))
    log_dens = kde.score_samples(domain)
    return np.exp(log_dens)

#%% md
#
# First, for the sake of this example, we generate fake data from a gaussian distribution with mean 10 and standard
# deviation 1.

#%%

np.random.seed(100)
mu, sigma = 10, 1  # true mean and standard deviation
data_1 = np.random.normal(mu, sigma, 100).reshape((-1, 1))
np.random.seed()

# plot the data and true distribution
count, bins, ignored = plt.hist(data_1, 30, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.title('data as histogram and true distribution to be estimated')
plt.show()


#%% md
#
# In a Bayesian setting, the definition of a prior pdf is a key point. The prior for the parameters must be defined in
# the model. Note that if no prior is given, an improper, uninformative, prior is chosen, :math:`p(\theta)=1` for all
# :math:`\theta`.

#%%

p0 = Uniform(loc=0., scale=15)
p1 = Lognormal(s=1., loc=0., scale=1.)
prior = JointIndependent(marginals=[p0, p1])

candidate_model = DistributionModel(distributions=Normal(loc=None, scale=None), n_parameters=2, prior=prior)

# Learn the unknown parameters using MCMC
from UQpy.sampling import MetropolisHastings

mh1 = MetropolisHastings(jump=10, burn_length=10, seed=[1.0, 0.2], random_state=123)

bayes_estimator = BayesParameterEstimation(inference_model=candidate_model,
                                           data=data_1,
                                           nsamples=500,
                                           sampling_class=mh1)

# print results
s = bayes_estimator.sampler.samples
plt.scatter(s[:, 0], s[:, 1])
plt.scatter(10, 1, marker='+', label='true parameter')
plt.title('MCMC samples')
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

domain = np.linspace(0, 15, 200)[:, np.newaxis]
pdf = pdf_from_kde(domain, s[:, 0])
ax[0].plot(domain, p0.pdf(domain), label='prior')
ax[0].plot(domain, pdf, label='posterior')
ax[0].set_title('posterior pdf of theta=mu')
ax[0].legend()

domain = np.linspace(0, 2, 200)[:, np.newaxis]
pdf = pdf_from_kde(domain, s[:, 1])
ax[1].plot(domain, p1.pdf(domain), label='prior')
ax[1].plot(domain, pdf, label='posterior')
ax[1].set_title('posterior pdf of theta=sigma')
ax[1].legend()

plt.show()