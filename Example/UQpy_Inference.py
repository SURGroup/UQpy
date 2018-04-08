from UQpyLibraries.Inference import *
import random

random.seed(31415926)

# data = [11624, 9388, 9471, 8927, 10865, 7698, 11744, 9238, 10319, 9750, 11462, 7939]
mu, sigma = 2, 1
data = np.random.normal(mu, sigma, 100)
#print(len(data))

model = 'normal'

### Information model selection using AIC and BIC
Information_MS = InforMS(data=data, model=model)

AIC_value = Information_MS.AIC
BIC_value = Information_MS.BIC
print('AIC:', AIC_value)
print('BIC:', BIC_value)

# ### Bayesian inference
BI = Bayes_Inference(data=data, model=model)
samples = BI.samples
Bayesian_evi = BI.Bayes_factor
print('Bayesian evidence:', Bayesian_evi)

## multimodel information inference
#normal, lognormal, gamma, inverse gaussian, logistic, cauchy, exponential
model = ['normal', 'cauchy', 'exponential']
# model = ['normal','cauchy', 'exponential', 'lognormal', 'invgauss', 'logistic', 'gamma']
MMI = MultiMI(data=data, model=model)
print('model', MMI.model_sort)
print('AICc_value', np.transpose(MMI.AICC))
print('AICc_delta', np.transpose(MMI.delta))
print('AICc_weights', np.transpose(MMI.weights))

### multimodel Bayesian model selection
model = ['normal', 'cauchy', 'exponential']
MBayesI = MultiMBayesI(data=data, model=model)
print('model', MBayesI.model_sort)
print('Multimodel_Bayesian_Inference_weights', MBayesI.weights)
# print('MLE_Bayes', MBayesI.mle_Bayes)

########################################################################################################################
########################################################################################################################
#   Test emcee

import emcee
import corner

mu, sigma = 0, 1
data = np.random.normal(mu, sigma, 100)
print(data)


def lnlike(data, theta):
    loglike = np.sum(stats.norm.logpdf(data, loc=theta[0],scale=theta[1]))
    return loglike

def lnprior(theta):
    m,s = theta
    if -5.0 < m < 5.0 and 0.0 < s < 5.0:
        return 0.0
    return -np.inf

def lnprob(theta, data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp+lnlike(data,theta)


nwalkers = 50
ndim = 2
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
sampler.run_mcmc(p0, 1000)
samples = sampler.chain[:, 10:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=["$m$", "$s$"],
                      truths=[0, 1, np.log(1)])
fig.savefig("triangle.png")


# def lnprob(x):
#     return -0.5 * np.sum(2 * x ** 2)
#
# ndim, nwalkers = 1, 10
# p0 = [np.random.rand(ndim) for i in range(nwalkers)]
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
# pos, prob, state = sampler.run_mcmc(p0, 20)