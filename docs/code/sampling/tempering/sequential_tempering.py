"""

Sequential Tempering for Bayesian Inference and Reliability analyses
====================================================================
"""

# %% md
# The general framework: one wants to sample from a distribution of the form
#
# .. math:: p_1 \left( x \right) = \frac{q_1 \left( x \right)p_0 \left( x \right)}{Z_1}
#
#
# where :math:`q_1 \left( x \right)` and :math:`p_0 \left( x \right)` can be evaluated; and potentially estimate the
# constant :math:`Z_1 = \int q_1 \left( x \right)p_0 \left( x \right) dx`.
#
# Sequential tempering introduces a sequence of intermediate distributions:
#
# .. math:: p_{\beta_j} \left( x \right) \propto q \left( x, \beta_j \right)p_0 \left( x \right)
#
# for values of :math:`\beta_j` in :math:`[0, 1]`. The algorithm starts with :math:`\beta_0 = 0`, which samples
# from the reference distribution :math:`p_0`, and ends for some :math:`j = m` such that :math:`\beta_m = 1`, sampling
# from the target. First, a set of sample points is generated from :math:`p_0 = p_{\beta_0}`, and then these are
# resampled according to some weights :math:`w_0` such that after resampling the points follow :math:`p_{\beta_1}`.
# This procedure of resampling is carried out at each intermediate level :math:`j` - resampling the points distributed
# as :math:`p_{\beta_{j}}` according to weights :math:`w_{j}` such that after resampling, the points are distributed
# according to :math:`p_{\beta_{j+1}}`. As the points are sequentially resampled to follow each intermediate
# distribution, eventually they are resampled from :math:`p_{\beta_{m-1}}` to follow :math:`p_{\beta_{m}} = p_1`.
#
# The weights are calculated as
#
# .. math:: w_j = \frac{q \left( x, \beta_{j+1} \right)}{q \left( x, \beta_j \right)}
#
# The normalizing constant is calculated during the generation of samples, as
#
# .. math:: Z_1 = \prod_{j = 0}^{m-1} \left\{ \frac{\sum_{i = 1}^{N_j} w_j}{N_j} \right\}
#
# where :math:`N_j` is the number of sample points generated from the intermediate distribution :math:`p_{\beta_j}`.

# %%

# %% md
# Bayesian Inference
# -------------------
#
# In the Bayesian setting, :math:`p_0` is the prior, and :math:`q \left( x, \beta_j \right) = \mathcal{L}\left( data, x \right) ^{\beta_j}`

# %%

from UQpy.run_model import RunModel, PythonModel
import numpy as np
from UQpy.distributions import Uniform, Normal, JointIndependent, MultivariateNormal
from UQpy.sampling import SequentialTemperingMCMC
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, uniform
from UQpy.sampling.mcmc import *


# %% md
#

# %%

def likelihood(x, b):
    mu1 = np.array([1., 1.])
    mu2 = -0.8 * np.ones(2)
    w1 = 0.5
    # Width of 0.1 in each dimension
    sigma1 = np.diag([0.02, 0.05])
    sigma2 = np.diag([0.05, 0.02])

    # Posterior is a mixture of two gaussians
    like = np.exp(np.logaddexp(np.log(w1) + multivariate_normal.logpdf(x=x, mean=mu1, cov=sigma1),
                               np.log(1. - w1) + multivariate_normal.logpdf(x=x, mean=mu2, cov=sigma2)))
    return like ** b


prior = JointIndependent(marginals=[Uniform(loc=-2.0, scale=4.0), Uniform(loc=-2.0, scale=4.0)])


# %% md
#

# %%

# estimate evidence
def estimate_evidence_from_prior_samples(size):
    samples = -2. + 4 * np.random.uniform(size=size * 2).reshape((size, 2))
    return np.mean(likelihood(samples, 1.0))


def func_integration(x1, x2):
    x = np.array([x1, x2]).reshape((1, 2))
    return likelihood(x, 1.0) * (1. / 4) ** 2


def estimate_evidence_from_quadrature():
    from scipy.integrate import dblquad
    ev = dblquad(func=func_integration, a=-2, b=2, gfun=lambda x: -2, hfun=lambda x: 2)
    return ev


x = np.arange(-2, 2, 0.02)
y = np.arange(-2, 2, 0.02)
xx, yy = np.meshgrid(x, y)
z = likelihood(np.concatenate([xx.reshape((-1, 1)), yy.reshape((-1, 1))], axis=-1), 1.0)
h = plt.contourf(x, y, z.reshape(xx.shape))
plt.title('Likelihood')
plt.axis('equal')
plt.show()

# for nMC in [50000, 100000, 500000, 1000000]:
#    print('Evidence = {}'.format(estimate_evidence_from_prior_samples(nMC)))
print('Evidence computed analytically = {}'.format(estimate_evidence_from_quadrature()[0]))

# %% md
#

# %%
sampler = MetropolisHastings(dimension=2, n_chains=20)
test = SequentialTemperingMCMC(pdf_intermediate=likelihood,
                               distribution_reference=prior,
                               save_intermediate_samples=True,
                               percentage_resampling=10,
                               sampler=sampler,
                               nsamples=4000)

# %% md
#

# %%

print('Normalizing Constant = ' + str(test.evidence))
print('Tempering Parameters = ' + str(test.tempering_parameters))

plt.figure()
plt.scatter(test.intermediate_samples[0][:, 0], test.intermediate_samples[0][:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[0]))
plt.show()

plt.figure()
plt.scatter(test.intermediate_samples[2][:, 0], test.intermediate_samples[2][:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[2]))
plt.show()

plt.figure()
plt.scatter(test.samples[:, 0], test.samples[:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[-1]))
plt.show()

# %% md
#  Reliability
# -------------------
#
# In the reliability context, :math:`p_0` is the pdf of the parameters, and
#
# .. math::    q \left( x, \beta_j \right) = I_{\beta_j} \left( x \right) = \frac{1}{1 + \exp{\left( \frac{G \left( x \right)}{\frac{1}{\beta_j} - 1} \right)}}
#
# where :math:`G \left( x \right)` is the performance function, negative if the system fails, and :math:`I_{\beta_j} \left( x \right)` are smoothed versions of the indicator function.

# %%

from scipy.stats import norm


def indic_sigmoid(y, beta):
    return 1. / (1. + np.exp(y / (1. / beta - 1.)))


fig, ax = plt.subplots(figsize=(4, 3.5))
ys = np.linspace(-5, 5, 100)
for i, s in enumerate(1. / np.array([1.01, 1.25, 2., 4., 70.])):
    ax.plot(ys, indic_sigmoid(y=ys, beta=s), label=r'$\beta={:.2f}$'.format(s), color='blue', alpha=1. - i / 6)
ax.set_xlabel(r'$y=g(\theta)$', fontsize=13)
ax.set_ylabel(r'$q_{\beta}(\theta)=I_{\beta}(y)$', fontsize=13)
# ax.set_title(r'Smooth versions of the indicator function', fontsize=14)
ax.legend(fontsize=8.5)
plt.show()

# %% md
#

# %%

beta = 2  # Specified Reliability Index
rho = 0.7  # Specified Correlation
dim = 2  # Dimension

# Define the correlation matrix
C = np.ones((dim, dim)) * rho
np.fill_diagonal(C, 1)
print(C)

# Print information related to the true probability of failure
e, v = np.linalg.eig(np.asarray(C))
beff = np.sqrt(np.max(e)) * beta
print(beff)
from scipy.stats import norm

pf_true = norm.cdf(-beta)
print('True pf={}'.format(pf_true))


# %% md
#

# %%


def estimate_Pf_0(samples, model_values):
    mask = model_values <= 0
    return np.sum(mask) / len(mask)


model = RunModel(model=PythonModel(model_script='local_reliability_funcs.py', model_object_name="correlated_gaussian",
                                   b_eff=beff, d=dim))
samples = MultivariateNormal(mean=np.zeros((2,)), cov=np.array([[1, 0.7], [0.7, 1]])).rvs(nsamples=20000)
model.run(samples=samples, append_samples=False)
model_values = np.array(model.qoi_list)

print('Prob. failure (MC) = {}'.format(estimate_Pf_0(samples, model_values)))

fig, ax = plt.subplots(figsize=(4, 3.5))
mask = np.squeeze(model_values <= 0)
ax.scatter(samples[mask, 0], samples[mask, 1], color='red', label='fail', alpha=0.5, marker='d')
ax.scatter(samples[~mask, 0], samples[~mask, 1], color='blue', label='safe', alpha=0.5)
plt.axis('equal')
# plt.title('Failure domain for reliability problem', fontsize=14)
plt.xlabel(r'$\theta_{1}$', fontsize=13)
plt.ylabel(r'$\theta_{2}$', fontsize=13)
ax.legend(fontsize=13)
fig.tight_layout()
plt.show()


# %% md
#

# %%

def indic_sigmoid(y, b):
    return 1.0 / (1.0 + np.exp((y * b) / (1.0 - b)))


def factor_param(x, b):
    model.run(samples=x, append_samples=False)
    G_values = np.array(model.qoi_list)
    return np.squeeze(indic_sigmoid(G_values, b))


prior = MultivariateNormal(mean=np.zeros((2,)), cov=C)

sampler = MetropolisHastings(dimension=2, n_chains=20)
test = SequentialTemperingMCMC(pdf_intermediate=factor_param,
                               distribution_reference=prior,
                               save_intermediate_samples=True,
                               percentage_resampling=10,
                               random_state=960242069,
                               sampler=sampler,
                               nsamples=3000)


# %% md
#

# %%

print('Estimated Probability of Failure = ' + str(test.evidence))
print('Tempering Parameters = ' + str(test.tempering_parameters))

plt.figure()
plt.scatter(test.intermediate_samples[0][:, 0], test.intermediate_samples[0][:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[0]))
plt.show()

plt.figure()
plt.scatter(test.intermediate_samples[2][:, 0], test.intermediate_samples[2][:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[2]))
plt.show()

plt.figure()
plt.scatter(test.samples[:, 0], test.samples[:, 1])
plt.title(r'$\beta = $' + str(test.tempering_parameters[-1]))
plt.show()
