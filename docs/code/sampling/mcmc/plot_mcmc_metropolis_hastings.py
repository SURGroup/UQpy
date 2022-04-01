"""

Sampling Rosenbrock distribution using Metropolis-Hastings
============================================================

In this example, the Metropolis-Hastings is employed to generate samples from a Rosenbrock distribution.
The method illustrates various aspects of the UQpy :class:`.MCMC` class:

- various ways of defining the target pdf to sample from,
- definition of input parameters required by the algorithm (proposal_type and proposal_scale for :class:`.MetropolisHastings`),
- running several chains in parallel,
- call diagnostics functions.
"""

# %% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.MCMC` class from UQpy.

# %%

from UQpy.sampling import MetropolisHastings
import numpy as np
import matplotlib.pyplot as plt
import time


# %% md
#
# Explore various ways of defining the target pdf
# -----------------------------------------------
# Define the Rosenbrock probability density function up to a scale factor. Here the pdf is defined directly in the
# python script
#
# - define the Rosenbrock probability density function up to a scale factor, this function only takes as input parameter
# the point x where to compute the pdf,
# - define a pdf function that also takes as argument a set of parameters params,
# - define a function that computes the log pdf up to a constant.
# Alternatively, the pdf can be defined in an external file that defines a distribution and its :code:`pdf` or
# :code:`log_pdf` methods
# (Rosenbrock.py)

# %%

def rosenbrock_no_params(x):
    return np.exp(-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20)


def log_rosenbrock_with_param(x, p):
    return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / p


x = MetropolisHastings(dimension=2, pdf_target=rosenbrock_no_params, burn_length=500, jump=50,
                       n_chains=1, nsamples=500)
print(x.samples.shape)
plt.figure()
plt.plot(x.samples[:, 0], x.samples[:, 1], 'o', alpha=0.5)
plt.show()

plt.figure()
x = MetropolisHastings(dimension=2, pdf_target=log_rosenbrock_with_param, burn_length=500,
                       jump=50, n_chains=1, args_target=(20,),
                       nsamples=500)
plt.plot(x.samples[:, 0], x.samples[:, 1], 'o')
plt.show()

# %% md
#
# In the following, a custom Rosenbrock distribution is defined and its :code:`log_pdf` method is used.

# %%

from UQpy.distributions import DistributionND


class Rosenbrock(DistributionND):
    def __init__(self, p=20.):
        super().__init__(p=p)

    def pdf(self, x):
        return np.exp(-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / self.parameters['p'])

    def log_pdf(self, x):
        return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / self.parameters['p']


log_pdf_target = Rosenbrock(p=20.).log_pdf

x = MetropolisHastings(dimension=2, pdf_target=log_pdf_target, burn_length=500, jump=50,
                       n_chains=1, nsamples=500)

# %% md
#
# In the following, we show that if :code:`burn_length` is set to :math:`0`, the first sample is always the seed.

# %%

seed = [[0., -1.], [1., 1.]]

x = MetropolisHastings(dimension=2, log_pdf_target=log_pdf_target, burn_length=0, jump=50,
                       seed=seed, concatenate_chains=False,
                       nsamples=500)
print(x.samples.shape)
plt.plot(x.samples[:, 0, 0], x.samples[:, 0, 1], 'o', alpha=0.5)
plt.plot(x.samples[:, 1, 0], x.samples[:, 1, 1], 'o', alpha=0.5)
plt.show()

print(seed)
print(x.samples[0, :, :])

# %% md
#
# The algorithm-specific parameters for MetropolisHastings are proposal and proposal_is_symmetric
# -------------------------------------------------------------------------------
# The default proposal is standard normal (symmetric).

# %%

# Define a few proposals to try out
from UQpy.distributions import JointIndependent, Normal, Uniform

proposals = [JointIndependent([Normal(), Normal()]),
             JointIndependent([Uniform(loc=-0.5, scale=1.5), Uniform(loc=-0.5, scale=1.5)]),
             Normal()]

proposals_is_symmetric = [True, False, False]

fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
for i, (proposal, symm) in enumerate(zip(proposals, proposals_is_symmetric)):
    print(i)
    try:
        x = MetropolisHastings(dimension=2, burn_length=500, jump=100, log_pdf_target=log_pdf_target,
                               proposal=proposal, proposal_is_symmetric=symm, n_chains=1,
                               nsamples=1000)
        ax[i].plot(x.samples[:, 0], x.samples[:, 1], 'o')
    except ValueError as e:
        print(e)
        print('This last call fails because the proposal is in dimension 1, while the target distribution is'
              ' in dimension 2')
plt.show()

# %% md
#
# Run several chains in parallel
# -------------------------------------------------------------------------------
# The user can provide the total number of samples :code:`nsamples`, or the number of samples per chain
# :code:`nsamples_per_chain`.

# %%

x = MetropolisHastings(dimension=2, log_pdf_target=log_pdf_target, jump=1000, burn_length=500,
                       seed=[[0., 0.], [1., 1.]], concatenate_chains=False,
                       nsamples=100)
plt.plot(x.samples[:, 0, 0], x.samples[:, 0, 1], 'o', label='chain 1', alpha=0.5)
plt.plot(x.samples[:, 1, 0], x.samples[:, 1, 1], 'o', label='chain 2', alpha=0.5)
print(x.samples.shape)
plt.legend()
plt.show()

x = MetropolisHastings(dimension=2, log_pdf_target=log_pdf_target, jump=1000, burn_length=500,
                       seed=[[0., 0.], [1., 1.]], concatenate_chains=False,
                       nsamples_per_chain=100)
plt.plot(x.samples[:, 0, 0], x.samples[:, 0, 1], 'o', label='chain 1', alpha=0.5)
plt.plot(x.samples[:, 1, 0], x.samples[:, 1, 1], 'o', label='chain 2', alpha=0.5)
print(x.samples.shape)
plt.legend()
plt.show()

# %% md
#
# Initialize without nsamples... then call run
# -------------------------------------------------------------------------------

# %%

t = time.time()

x = MetropolisHastings(dimension=2, log_pdf_target=log_pdf_target, jump=1000,
                       burn_length=500, seed=[[0., 0.], [1., 1.]], concatenate_chains=False)
print('Elapsed time for initialization: {} s'.format(time.time() - t))

t = time.time()
x.run(nsamples=100)
print('Elapsed time for running MCMC: {} s'.format(time.time() - t))
print('nburn, jump at first run: {}, {}'.format(x.burn_length, x.jump))
print('total nb of samples: {}'.format(x.samples.shape[0]))

plt.plot(x.samples[:, 0, 0], x.samples[:, 0, 1], 'o', label='chain 1')
plt.plot(x.samples[:, 1, 0], x.samples[:, 1, 1], 'o', label='chain 2')
plt.legend()
plt.show()

# %% md
#
# Run another example with a bivariate distributon with copula dependence - use random_state to always have the
# same outputs
# ----------------------------------------------------------------------------------------------------------------------

# %%

from UQpy.distributions import Normal, Gumbel, JointCopula, JointIndependent

dist_true = JointCopula(marginals=[Normal(), Normal()], copula=Gumbel(theta=2.))
proposal = JointIndependent(marginals=[Normal(scale=0.2), Normal(scale=0.2)])

sampler = MetropolisHastings(dimension=2, log_pdf_target=dist_true.log_pdf, proposal=proposal,
                             seed=[0., 0.], random_state=123,
                             nsamples=500)
print(sampler.samples.shape)
print(np.round(sampler.samples[-5:], 4))

plt.plot(sampler.samples[:, 0], sampler.samples[:, 1], linestyle='none', marker='+')
