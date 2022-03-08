"""

Parameter estimation using MCMC - Regression Model
=============================================================================

Here a model is defined that is of the form

.. math:: y=f(\theta) + \epsilon

where f consists in running RunModel. In particular, here :math:`f(\theta)=\theta_{0} x + \theta_{1} x^{2}` is a
regression model.

"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

import numpy as np
import matplotlib.pyplot as plt

from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings
from UQpy.inference.inference_models.ComputationalModel import ComputationalModel
from UQpy.RunModel import RunModel
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
# First we generate synthetic data, and add some noise to it.

#%%

# Generate data
param_true = np.array([1.0, 2.0]).reshape((1, -1))
print('Shape of true parameter vector: {}'.format(param_true.shape))

h_func = RunModel(model_script='pfn_models.py', model_object_name='model_quadratic', vec=False,
                  var_names=['theta_0', 'theta_1'])
h_func.run(samples=param_true)
data_clean = np.array(h_func.qoi_list[0])
print(data_clean.shape)

# Add noise, use a RandomState for reproducible results
error_covariance = 1.
noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50, random_state=123).reshape((50,))
data_3 = data_clean + noise
print('Shape of data: {}'.format(data_3.shape))
print(data_3[:4])

p0 = Normal()
p1 = Normal()
prior = JointIndependent(marginals=[p0, p1])

inference_model = ComputationalModel(n_parameters=2, runmodel_object=h_func, error_covariance=error_covariance,
                                     prior=prior)

proposal = JointIndependent([Normal(scale=0.1), Normal(scale=0.05)])

mh1 = MetropolisHastings(jump=10, burn_length=0, proposal=proposal, seed=[0.5, 2.5],
                         random_state=456)

bayes_estimator = BayesParameterEstimation(inference_model=inference_model,
                                           data=data_3,
                                           sampling_class=mh1,
                                           nsamples=500)

s = bayes_estimator.sampler.samples
plt.scatter(s[:, 0], s[:, 1])
plt.scatter(1.0, 2.0, label='true value')
plt.title('MCMC samples')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

domain = np.linspace(-4, 4, 200)[:, np.newaxis]
pdf_ = pdf_from_kde(domain, s[:, 0])
ax[0].plot(domain, pdf_, label='posterior')
ax[0].plot(domain, p0.pdf(domain), label='prior')
ax[0].set_title('posterior pdf of theta_{1}')

domain = np.linspace(-4, 4, 200)[:, np.newaxis]
pdf_ = pdf_from_kde(domain, s[:, 1])
ax[1].plot(domain, pdf_, label='posterior')
ax[1].plot(domain, p1.pdf(domain), label='prior')
ax[1].set_title('posterior pdf of theta_{2}')

plt.show()

print(bayes_estimator.sampler.samples[:4])

shutil.rmtree(h_func.model_dir)
