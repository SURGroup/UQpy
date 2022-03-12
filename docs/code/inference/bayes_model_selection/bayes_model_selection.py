"""

Bayes Model Selection - Regression Models
=============================================================================

In the following we present an example for which the posterior pdf of the parameters, evidences and model probabilities
can be computed analytically. We drop the :math:`m_{j}` subscript when referring to model parameters for simplicity. Three
models are considered (the domain :math:`x` is fixed and consists in 50 equally spaced points):

.. math:: m_{linear}: \quad y = θ_{0} x + \epsilon

.. math:: m_{quadratic}: \quad y = θ_{0} x + θ_{1} x^2 + \epsilon

.. math:: m_{cubic}: \quad y = θ_{0} x + θ_{1} x^2+ θ_{2} x^3 + \epsilon

All three models can be written in a compact form as :math:`y=X θ + \epsilon`, where :math:`X` contains the
necessary powers of :math:`x`. For all three models, the prior is chosen to be Gaussian,
:math:`p(θ) = N(\cdot, θ_{prior}, \Sigma_{prior})`, and so is the noise
:math:`\epsilon \sim N(\cdot; 0, \sigma_{n}^{2} I)`.
"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

import numpy as np
import matplotlib.pyplot as plt
from UQpy.sampling.mcmc import MetropolisHastings
from UQpy.inference import BayesModelSelection, BayesParameterEstimation, ComputationalModel
from UQpy.RunModel import RunModel  # required to run the quadratic model
from UQpy.distributions import Normal, JointIndependent
from scipy.stats import multivariate_normal, norm


#%% md
#
# Generate data from a quadratic function

#%%

param_true = np.array([1.0, 2.0]).reshape(1, -1)
var_n = 1
error_covariance = var_n * np.eye(50)
print(param_true.shape)

z = RunModel(samples=param_true, model_script='local_pfn_models.py', model_object_name='model_quadratic', vec=False,
             var_names=['theta_1', 'theta_2'])
data_clean = z.qoi_list[0].reshape((-1,))
data = data_clean + Normal(scale=np.sqrt(var_n)).rvs(nsamples=data_clean.size, random_state=456).reshape((-1,))

#%% md
#
# Define the models, compute the true values of the evidence.
#
# For all three models, a Gaussian prior is chosen for the parameters, with mean and covariance matrix of the
# appropriate dimensions. Each model is given prior probability :math:`P(m_{j}) = 1/3`.

#%%

model_names = ['model_linear', 'model_quadratic', 'model_cubic']
model_n_params = [1, 2, 3]
model_prior_means = [[0.], [0., 0.], [0., 0., 0.]]
model_prior_stds = [[10.], [1., 1.], [1., 2., 0.25]]


evidences = []
model_posterior_means = []
model_posterior_stds = []
for n, model in enumerate(model_names):
    # compute matrix X
    X = np.linspace(0, 10, 50).reshape((-1, 1))
    if n == 1:  # quadratic model
        X = np.concatenate([X, X ** 2], axis=1)
    if n == 2:  # cubic model
        X = np.concatenate([X, X ** 2, X ** 3], axis=1)

    # compute posterior pdf
    m_prior = np.array(model_prior_means[n]).reshape((-1, 1))
    S_prior = np.diag(np.array(model_prior_stds[n]) ** 2)
    S_posterior = np.linalg.inv(1 / var_n * np.matmul(X.T, X) + np.linalg.inv(S_prior))
    m_posterior = np.matmul(S_posterior,
                            1 / var_n * np.matmul(X.T, data.reshape((-1, 1))) + np.matmul(np.linalg.inv(S_prior),
                                                                                          m_prior))
    m_prior = m_prior.reshape((-1,))
    m_posterior = m_posterior.reshape((-1,))
    model_posterior_means.append(list(m_posterior))
    model_posterior_stds.append(list(np.sqrt(np.diag(S_posterior))))
    print('posterior mean and covariance for ' + model)
    print(m_posterior, S_posterior)

    # compute evidence, evaluate the formula at the posterior mean
    like_theta = multivariate_normal.pdf(data, mean=np.matmul(X, m_posterior).reshape((-1,)), cov=error_covariance)
    prior_theta = multivariate_normal.pdf(m_posterior, mean=m_prior, cov=S_prior)
    posterior_theta = multivariate_normal.pdf(m_posterior, mean=m_posterior, cov=S_posterior)
    evidence = like_theta * prior_theta / posterior_theta
    evidences.append(evidence)
    print('evidence for ' + model + '= {}\n'.format(evidence))

# compute the posterior probability of each model
tmp = [1 / 3 * evidence for evidence in evidences]
model_posterior_probas = [p / sum(tmp) for p in tmp]

print('posterior probabilities of all three models')
print(model_posterior_probas)

#%% md
#
# Define the models for use in UQpy

#%%

candidate_models = []
for n, model_name in enumerate(model_names):
    run_model = RunModel(model_script='local_pfn_models.py', model_object_name=model_name, vec=False)
    prior = JointIndependent([Normal(loc=m, scale=std) for m, std in
                              zip(model_prior_means[n], model_prior_stds[n])])
    model = ComputationalModel(n_parameters=model_n_params[n],
                               runmodel_object=run_model, prior=prior,
                               error_covariance=error_covariance,
                               name=model_name)
    candidate_models.append(model)


#%% md
#
# Run MCMC for one model

#%%

# Quadratic model
sampling = MetropolisHastings(args_target=(data, ),
                              log_pdf_target=candidate_models[1].evaluate_log_posterior,
                              jump=10, burn_length=100,
                              proposal=JointIndependent([Normal(scale=0.1), ] * 2),
                              seed=[0., 0.], random_state=123)

bayesMCMC = BayesParameterEstimation(sampling_class=sampling,
                                     inference_model=candidate_models[1],
                                     data=data,
                                     nsamples=3500)

# plot prior, true posterior and estimated posterior
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for n_p in range(2):
    domain_plot = np.linspace(-0.5, 3, 200)
    ax[n_p].plot(domain_plot, norm.pdf(domain_plot, loc=model_prior_means[1][n_p], scale=model_prior_stds[1][n_p]),
                 label='prior', color='green', linestyle='--')
    ax[n_p].plot(domain_plot, norm.pdf(domain_plot, loc=model_posterior_means[1][n_p],
                                       scale=model_posterior_stds[1][n_p]),
                 label='true posterior', color='red', linestyle='-')
    ax[n_p].hist(bayesMCMC.sampler.samples[:, n_p], density=True, bins=30, label='estimated posterior MCMC')
    ax[n_p].legend()
    ax[n_p].set_title('MCMC for quadratic model')
plt.show()


#%% md
#
# Run Bayesian Model Selection for all three models

#%%

proposals = [Normal(scale=0.1),
             JointIndependent([Normal(scale=0.1), Normal(scale=0.1)]),
             JointIndependent([Normal(scale=0.15), Normal(scale=0.1), Normal(scale=0.05)])]
nsamples = [2000, 6000, 14000]
nburn = [500, 2000, 4000]
jump = [5, 10, 25]


sampling_inputs=list()
estimators = []
for i in range(3):
    sampling = MetropolisHastings(args_target=(data, ),
                                  log_pdf_target=candidate_models[i].evaluate_log_posterior,
                                  jump=jump[i],
                                  burn_length=nburn[i],
                                  proposal=proposals[i],
                                  seed=model_prior_means[i],
                                  random_state=123)
    estimators.append(BayesParameterEstimation(inference_model=candidate_models[i], data=data,
                                                  sampling_class=sampling))

selection = BayesModelSelection(parameter_estimators=estimators,
                                data=data,
                                prior_probabilities=[1. / 3., 1. / 3., 1. / 3.],
                                nsamples=nsamples)

sorted_indices = np.argsort(selection.probabilities)[::-1]
print('Sorted models:')
print([selection.candidate_models[i].name for i in sorted_indices])
print('Evidence of sorted models:')
print([selection.evidences[i] for i in sorted_indices])
print('Posterior probabilities of sorted models:')
print([selection.probabilities[i] for i in sorted_indices])

#%% md
#
# As of version 2, the implementation of BayesModelSelection in UQpy uses the method of the harmonic mean to compute
# the models' evidence. This method is known to behave quite poorly, in particular it yeidls estimates with large
# variance. In the problem above, this implementation does not consistently detects that the quadratic model has the
# highest model probability. Future versions of UQpy will integrate more advanced methods for the estimation of the
# evidence.

#%%

for i, (m, be) in enumerate(zip(selection.candidate_models, selection.bayes_estimators)):
    # plot prior, true posterior and estimated posterior
    print('Posterior parameters for model ' + m.name)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for n_p in range(m.n_parameters):
        domain_plot = np.linspace(min(be.sampler.samples[:, n_p]), max(be.sampler.samples[:, n_p]), 200)
        ax[n_p].plot(domain_plot, norm.pdf(domain_plot, loc=model_prior_means[i][n_p],
                                           scale=model_prior_stds[i][n_p]),
                     label='prior', color='green', linestyle='--')
        ax[n_p].plot(domain_plot, norm.pdf(domain_plot, loc=model_posterior_means[i][n_p],
                                           scale=model_posterior_stds[i][n_p]),
                     label='true posterior', color='red', linestyle='-')
        ax[n_p].hist(be.sampler.samples[:, n_p], density=True, bins=30, label='estimated posterior MCMC')
        ax[n_p].legend()
    plt.show()

shutil.rmtree(z.model_dir)
shutil.rmtree(candidate_models[0].runmodel_object.model_dir)
shutil.rmtree(candidate_models[1].runmodel_object.model_dir)
shutil.rmtree(candidate_models[2].runmodel_object.model_dir)


